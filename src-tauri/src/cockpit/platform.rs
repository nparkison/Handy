//! Platform probes for Swap last: the foreground window, real (non-injected)
//! user input since the paste, and held modifier keys.
//!
//! Only Windows implements them. Elsewhere every probe reports "unknown", so
//! the swap guards fall back to copying the other version.

use std::time::{Duration, Instant};

/// The window that had focus when Handy pasted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForegroundApp {
    /// Opaque window handle.
    pub window: isize,
    /// Full path of the owning process image, when it could be queried.
    pub process_path: Option<String>,
}

/// Input that happened within this long before a swap trigger press is
/// attributed to the trigger itself (hook delivery order is not guaranteed).
pub const TRIGGER_SLACK: Duration = Duration::from_millis(50);

pub use imp::{foreground_app, modifiers_released, real_input_between, start_input_watch};

#[cfg(windows)]
mod imp {
    use super::ForegroundApp;
    use log::{debug, warn};
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, Once};
    use std::time::{Duration, Instant};
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, HINSTANCE, LPARAM, LRESULT, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, GetForegroundWindow, GetMessageW,
        GetWindowThreadProcessId, SetWindowsHookExW, KBDLLHOOKSTRUCT, LLKHF_INJECTED,
        LLMHF_INJECTED, MSG, MSLLHOOKSTRUCT, WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN,
        WM_LBUTTONDOWN, WM_MBUTTONDOWN, WM_RBUTTONDOWN, WM_SYSKEYDOWN, WM_XBUTTONDOWN,
    };

    /// Recent qualifying inputs, newest last.
    static INPUTS: Mutex<VecDeque<Instant>> = Mutex::new(VecDeque::new());
    const MAX_INPUTS: usize = 64;
    static WATCHING: AtomicBool = AtomicBool::new(false);
    static START: Once = Once::new();

    /// Shift, Ctrl, Alt, Win (generic and left/right variants).
    const MODIFIER_VKS: &[u32] = &[
        0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
    ];

    fn record_input() {
        let mut inputs = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
        if inputs.len() == MAX_INPUTS {
            inputs.pop_front();
        }
        inputs.push_back(Instant::now());
    }

    unsafe extern "system" fn keyboard_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let message = wparam.0 as u32;
            if message == WM_KEYDOWN || message == WM_SYSKEYDOWN {
                let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
                let injected = info.flags.0 & LLKHF_INJECTED.0 != 0;
                // Modifiers alone never change text; the swap shortcut's own
                // modifiers are pressed before it.
                if !injected && !MODIFIER_VKS.contains(&info.vkCode) {
                    record_input();
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    unsafe extern "system" fn mouse_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let message = wparam.0 as u32;
            // Button presses can move the caret; movement and wheel cannot.
            if matches!(
                message,
                WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_XBUTTONDOWN
            ) {
                let info = &*(lparam.0 as *const MSLLHOOKSTRUCT);
                if info.flags & LLMHF_INJECTED == 0 {
                    record_input();
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    /// Install low-level keyboard/mouse hooks on a dedicated thread (once).
    /// The callbacks only timestamp key and button presses; they never block
    /// or consume input.
    pub fn start_input_watch() {
        START.call_once(|| {
            let spawned = std::thread::Builder::new()
                .name("handy-input-watch".into())
                .spawn(|| unsafe {
                    let module = GetModuleHandleW(PCWSTR::null())
                        .ok()
                        .map(|m| HINSTANCE(m.0));
                    let keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), module, 0);
                    let mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), module, 0);
                    match (keyboard, mouse) {
                        (Ok(_), Ok(_)) => {
                            WATCHING.store(true, Ordering::Release);
                            debug!("Input watch started");
                        }
                        (k, m) => {
                            warn!(
                                "Input watch unavailable (keyboard: {:?}, mouse: {:?}); swaps will copy instead",
                                k.err(),
                                m.err()
                            );
                            return;
                        }
                    }
                    // Low-level hooks are serviced by this thread's message loop.
                    let mut msg = MSG::default();
                    while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                        let _ = DispatchMessageW(&msg);
                    }
                    WATCHING.store(false, Ordering::Release);
                });
            if let Err(e) = spawned {
                warn!("Failed to start input watch thread: {e}");
            }
        });
    }

    /// Was there real key or mouse-button input in `(from, until)`?
    /// `None` when the watch is not running.
    pub fn real_input_between(from: Instant, until: Instant) -> Option<bool> {
        if !WATCHING.load(Ordering::Acquire) {
            return None;
        }
        let inputs = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
        Some(inputs.iter().any(|&at| at > from && at < until))
    }

    pub fn foreground_app() -> Option<ForegroundApp> {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.0.is_null() {
                return None;
            }
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            Some(ForegroundApp {
                window: hwnd.0 as isize,
                process_path: process_path(pid),
            })
        }
    }

    unsafe fn process_path(pid: u32) -> Option<String> {
        if pid == 0 {
            return None;
        }
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buffer = [0u16; 1024];
        let mut len = buffer.len() as u32;
        let result = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut len,
        );
        let _ = CloseHandle(handle);
        result.ok()?;
        Some(String::from_utf16_lossy(&buffer[..len as usize]))
    }

    fn any_modifier_down() -> bool {
        MODIFIER_VKS
            .iter()
            .any(|&vk| unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0)
    }

    /// Wait up to `timeout` for Shift/Ctrl/Alt/Win to be released. Injecting
    /// Shift+Left while Ctrl is held would select whole words.
    pub fn modifiers_released(timeout: Duration) -> bool {
        let start = Instant::now();
        loop {
            if !any_modifier_down() {
                return true;
            }
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::ForegroundApp;
    use std::time::{Duration, Instant};

    pub fn start_input_watch() {}

    pub fn real_input_between(_from: Instant, _until: Instant) -> Option<bool> {
        None
    }

    pub fn foreground_app() -> Option<ForegroundApp> {
        None
    }

    pub fn modifiers_released(_timeout: Duration) -> bool {
        true
    }
}

/// `true` when real input happened after the paste and before the trigger
/// (minus [`TRIGGER_SLACK`]); `None` when it cannot be known.
pub fn input_since_paste(pasted_at: Instant, trigger_at: Instant) -> Option<bool> {
    let until = trigger_at.checked_sub(TRIGGER_SLACK).unwrap_or(trigger_at);
    if until <= pasted_at {
        return real_input_between(pasted_at, pasted_at);
    }
    real_input_between(pasted_at, until)
}
