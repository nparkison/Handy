//! Platform probes for Swap last: the foreground window, real user input
//! since the paste, and held modifier keys.
//!
//! Only Windows implements them. Elsewhere every probe reports "unknown", so
//! the swap guards fall back to copying the other version.
//!
//! ## Input watch (Windows)
//! Low-level keyboard/mouse hooks timestamp every key and mouse-button press
//! into a lock-free ring. They are installed only while Swap last is reachable
//! (a Swap last binding or tap gestures) and removed when it stops being
//! reachable. Every event counts as user input, injected ones included (the
//! on-screen keyboard, AutoHotkey and KVM software inject), except Handy's own
//! injections (enigo events carrying [`crate::input::INJECTION_MARKER`], the
//! hotkey listener's "menu mask" key, the liveness probe). Windows silently
//! removes a hook that is too slow (`LowLevelHooksTimeout`), so a swap first
//! sends a probe event and only trusts the watch when the hook sees it; when
//! it does not, the hooks are re-installed, so the next paste is covered
//! again.

use std::time::{Duration, Instant};

/// The window that had focus when Handy pasted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForegroundApp {
    /// Opaque window handle.
    pub window: isize,
    /// Full path of the owning process image, when it could be queried.
    pub process_path: Option<String>,
}

/// The trigger's own key/button press may be timestamped by the hook slightly
/// before or after the trigger instant Handy recorded. Only presses of that
/// same key/button within this window are attributed to the trigger.
pub const TRIGGER_SLACK: Duration = Duration::from_millis(50);

/// Ring capacity of recorded presses.
pub(crate) const INPUT_RING: usize = 128;

/// Input code of a key (its virtual-key code) or a mouse button.
pub(crate) type InputCode = u16;
pub(crate) const MOUSE_LEFT: InputCode = 0x101;
#[cfg(windows)]
const MOUSE_RIGHT: InputCode = 0x102;
#[cfg(windows)]
const MOUSE_MIDDLE: InputCode = 0x103;
#[cfg(windows)]
const MOUSE_X: InputCode = 0x104; // + 0 / 1 for XBUTTON1 / XBUTTON2

/// Unassigned virtual key used both by the liveness probe and by the
/// hotkey listener's (handy-keys) "menu mask" on Win/Alt hotkeys. Never real
/// typing, so it is never recorded.
pub(crate) const UNASSIGNED_VK: u32 = 0xE8;
/// `dwExtraInfo` of handy-keys' injected menu-mask events.
pub(crate) const HANDY_KEYS_MARKER: usize = 0x484B_4D4D;
/// Shift, Ctrl, Alt, Win (generic and left/right variants).
pub(crate) const MODIFIER_VKS: &[u32] = &[
    0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
];

/// Whether a key-down seen by the keyboard hook is user input. Modifiers
/// alone never change text (and the swap shortcut's own modifiers are pressed
/// before it); Handy's own injections are not user input; anything else
/// injected is (on-screen keyboards, AutoHotkey, KVM software).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn is_user_key_press(vk: u32, extra_info: usize) -> bool {
    vk != UNASSIGNED_VK
        && !MODIFIER_VKS.contains(&vk)
        && extra_info != crate::input::INJECTION_MARKER
        && extra_info != HANDY_KEYS_MARKER
}

/// Whether a mouse-button press seen by the mouse hook is user input.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn is_user_mouse_press(extra_info: usize) -> bool {
    extra_info != crate::input::INJECTION_MARKER
}

pub use imp::{
    clipboard_sequence, foreground_app, last_press_of, modifiers_released, recorded_inputs,
    start_input_watch, stop_input_watch, watch_verified_since, window_title, ClipboardBackup,
};

/// Pure decision behind [`input_since_paste`], on microsecond timestamps.
///
/// `events` are recorded presses `(time_us, code)` (any order). Real input is
/// any press in `(from, until]`, except presses of the trigger's own key or
/// button at or after `trigger - slack` (the trigger itself, a held trigger's
/// auto-repeat, the second tap of a double-tap). The trigger's code is the
/// press closest to `trigger` within `slack`. When the ring holds nothing but
/// presses newer than `from`, older presses may have been overwritten, so
/// that counts as input too.
pub(crate) fn real_input_in(
    events: &[(u64, InputCode)],
    capacity: usize,
    from: u64,
    trigger: u64,
    until: u64,
    slack: u64,
) -> bool {
    if events.len() >= capacity && events.iter().all(|(t, _)| *t > from) {
        return true;
    }
    // A trigger before the paste (e.g. a queued swap) excuses nothing after it.
    if trigger < from {
        return events.iter().any(|&(t, _)| t > from && t <= until);
    }
    let window_start = trigger.saturating_sub(slack);
    let trigger_code = events
        .iter()
        .filter(|(t, _)| *t >= window_start && *t <= trigger.saturating_add(slack))
        .min_by_key(|(t, _)| t.abs_diff(trigger))
        .map(|&(_, code)| code);
    events.iter().any(|&(t, code)| {
        t > from && t <= until && !(Some(code) == trigger_code && t >= window_start)
    })
}

/// `Some(true)` when real input happened after the paste and up to `until`
/// (other than the trigger's own key/button); `None` when it cannot be known
/// (no input watch since before the paste). With `probe`, the hook is also
/// verified to still be alive (blocks up to ~150 ms).
pub fn input_since_paste(
    pasted_at: Instant,
    trigger_at: Instant,
    until: Instant,
    probe: bool,
) -> Option<bool> {
    if !watch_verified_since(pasted_at, probe) {
        return None;
    }
    let (events, base) = recorded_inputs()?;
    let us = |at: Instant| at.saturating_duration_since(base).as_micros() as u64;
    Some(real_input_in(
        &events,
        INPUT_RING,
        us(pasted_at),
        us(trigger_at),
        us(until),
        TRIGGER_SLACK.as_micros() as u64,
    ))
}

#[cfg(windows)]
mod imp {
    use super::{
        is_user_key_press, is_user_mouse_press, ForegroundApp, InputCode, INPUT_RING, MODIFIER_VKS,
        MOUSE_LEFT, MOUSE_MIDDLE, MOUSE_RIGHT, MOUSE_X, UNASSIGNED_VK,
    };
    use log::{debug, warn};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, LRESULT, WPARAM};
    use windows::Win32::System::Threading::{
        GetCurrentProcessId, GetCurrentThread, GetCurrentThreadId, OpenProcess,
        QueryFullProcessImageNameW, SetThreadPriority, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION, THREAD_PRIORITY_TIME_CRITICAL,
    };
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
        VIRTUAL_KEY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, GetForegroundWindow, GetMessageW, GetWindowTextLengthW,
        GetWindowTextW, GetWindowThreadProcessId, PeekMessageW, PostThreadMessageW,
        SetWindowsHookExW, UnhookWindowsHookEx, KBDLLHOOKSTRUCT, MSG, MSLLHOOKSTRUCT, PM_NOREMOVE,
        WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN, WM_LBUTTONDOWN, WM_MBUTTONDOWN, WM_QUIT,
        WM_RBUTTONDOWN, WM_SYSKEYDOWN, WM_XBUTTONDOWN,
    };

    /// Marker of the liveness probe (a key-up of an unassigned virtual key).
    const PROBE_MARKER: usize = 0x4841_4E44; // "HAND"
    const PROBE_TIMEOUT: Duration = Duration::from_millis(150);
    /// After the hooks could not be installed, wait this long before trying
    /// again (on the next settings change or paste).
    const INSTALL_RETRY_AFTER: Duration = Duration::from_secs(30);

    /// Clock origin of every stored timestamp (microseconds since BASE).
    static BASE: OnceLock<Instant> = OnceLock::new();
    /// Recorded presses: `(time_us << 16) | code`, 0 = empty slot.
    static RING: [AtomicU64; INPUT_RING] = [const { AtomicU64::new(0) }; INPUT_RING];
    static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
    /// When the hooks of the running watch were installed (µs + 1; 0 = none).
    static WATCH_SINCE: AtomicU64 = AtomicU64::new(0);
    /// Last time the hook saw the liveness probe (µs + 1).
    static PROBE_SEEN: AtomicU64 = AtomicU64::new(0);

    /// The watch should run.
    static DESIRED: AtomicBool = AtomicBool::new(false);
    /// A hook thread exists (starting, running or stopping).
    static THREAD_ALIVE: AtomicBool = AtomicBool::new(false);
    /// The hook thread's id once its message queue exists (0 = not yet).
    static THREAD_ID: AtomicU32 = AtomicU32::new(0);
    /// Serialises start/stop decisions with the hook thread's exit decision.
    static CONTROL: Mutex<()> = Mutex::new(());
    /// When installing the hooks last failed (µs + 1; 0 = never): do not
    /// retry (and warn) on every paste, only after [`INSTALL_RETRY_AFTER`].
    static INSTALL_FAILED_AT: AtomicU64 = AtomicU64::new(0);

    fn base() -> Instant {
        *BASE.get_or_init(Instant::now)
    }

    fn now_us(base: Instant) -> u64 {
        Instant::now().saturating_duration_since(base).as_micros() as u64
    }

    /// Hook callback hot path: two atomic operations, no locks, no allocation.
    #[inline]
    fn record_press(code: InputCode) {
        let Some(base) = BASE.get() else { return };
        let slot = NEXT_SLOT.fetch_add(1, Ordering::Relaxed) % INPUT_RING;
        let packed = ((now_us(*base) + 1) << 16) | u64::from(code);
        RING[slot].store(packed, Ordering::Release);
    }

    unsafe extern "system" fn keyboard_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 && lparam.0 != 0 {
            let message = wparam.0 as u32;
            // SAFETY: for HC_ACTION, lparam points to a KBDLLHOOKSTRUCT.
            let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
            if info.dwExtraInfo == PROBE_MARKER {
                if let Some(base) = BASE.get() {
                    PROBE_SEEN.store(now_us(*base) + 1, Ordering::Release);
                }
                // Invisible to everyone else.
                return LRESULT(1);
            }
            if (message == WM_KEYDOWN || message == WM_SYSKEYDOWN)
                && is_user_key_press(info.vkCode, info.dwExtraInfo)
            {
                record_press(info.vkCode.min(0xFF) as InputCode);
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    unsafe extern "system" fn mouse_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 && lparam.0 != 0 {
            // Button presses can move the caret; movement and wheel cannot.
            let button = match wparam.0 as u32 {
                WM_LBUTTONDOWN => Some(MOUSE_LEFT),
                WM_RBUTTONDOWN => Some(MOUSE_RIGHT),
                WM_MBUTTONDOWN => Some(MOUSE_MIDDLE),
                WM_XBUTTONDOWN => Some(MOUSE_X),
                _ => None,
            };
            if let Some(mut button) = button {
                // SAFETY: for HC_ACTION, lparam points to an MSLLHOOKSTRUCT.
                let info = &*(lparam.0 as *const MSLLHOOKSTRUCT);
                if is_user_mouse_press(info.dwExtraInfo) {
                    if button == MOUSE_X && (info.mouseData >> 16) == 2 {
                        button += 1;
                    }
                    record_press(button);
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    /// Body of the hook thread: install, pump until WM_QUIT, uninstall; then
    /// either exit or (when the watch was re-enabled meanwhile) go again.
    fn hook_thread() {
        unsafe {
            // Low-level hooks are serviced on this thread: keep it ahead of
            // a CPU-saturating transcription so input never waits on us.
            if let Err(e) = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) {
                debug!("Input watch: could not raise thread priority: {e}");
            }
            // Create the message queue before publishing the thread id, so a
            // WM_QUIT posted right after can never be lost.
            let mut msg = MSG::default();
            let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE);
            {
                let _guard = CONTROL.lock().unwrap_or_else(|e| e.into_inner());
                THREAD_ID.store(GetCurrentThreadId(), Ordering::SeqCst);
                if !DESIRED.load(Ordering::SeqCst) {
                    THREAD_ID.store(0, Ordering::SeqCst);
                    THREAD_ALIVE.store(false, Ordering::SeqCst);
                    return;
                }
            }
        }
        loop {
            run_hooks_once();
            let _guard = CONTROL.lock().unwrap_or_else(|e| e.into_inner());
            if !DESIRED.load(Ordering::SeqCst) {
                THREAD_ID.store(0, Ordering::SeqCst);
                THREAD_ALIVE.store(false, Ordering::SeqCst);
                debug!("Input watch stopped");
                return;
            }
        }
    }

    fn run_hooks_once() {
        unsafe {
            let base = base();
            let keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), None, 0);
            let mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), None, 0);
            let (keyboard, mouse) = match (keyboard, mouse) {
                (Ok(k), Ok(m)) => (k, m),
                (k, m) => {
                    warn!(
                        "Input watch unavailable (keyboard: {:?}, mouse: {:?}); swaps will copy instead",
                        k.as_ref().err(),
                        m.as_ref().err()
                    );
                    if let Ok(k) = k {
                        let _ = UnhookWindowsHookEx(k);
                    }
                    if let Ok(m) = m {
                        let _ = UnhookWindowsHookEx(m);
                    }
                    // Do not spin: retry only after INSTALL_RETRY_AFTER.
                    INSTALL_FAILED_AT.store(now_us(base) + 1, Ordering::SeqCst);
                    DESIRED.store(false, Ordering::SeqCst);
                    return;
                }
            };
            WATCH_SINCE.store(now_us(base) + 1, Ordering::SeqCst);
            debug!("Input watch started");
            let mut msg = MSG::default();
            loop {
                // 0 = WM_QUIT, -1 = error: both end the loop.
                let result = GetMessageW(&mut msg, None, 0, 0).0;
                if result == 0 || result == -1 {
                    if result == -1 {
                        warn!("Input watch: GetMessageW failed");
                    }
                    break;
                }
                let _ = DispatchMessageW(&msg);
            }
            WATCH_SINCE.store(0, Ordering::SeqCst);
            let _ = UnhookWindowsHookEx(keyboard);
            let _ = UnhookWindowsHookEx(mouse);
        }
    }

    /// Install the low-level hooks on their own thread (no-op when running).
    pub fn start_input_watch() {
        if DESIRED.load(Ordering::SeqCst) && THREAD_ALIVE.load(Ordering::SeqCst) {
            return;
        }
        let base = base();
        let failed_at = INSTALL_FAILED_AT.load(Ordering::SeqCst);
        if failed_at != 0
            && now_us(base).saturating_sub(failed_at - 1) < INSTALL_RETRY_AFTER.as_micros() as u64
        {
            return;
        }
        let _guard = CONTROL.lock().unwrap_or_else(|e| e.into_inner());
        DESIRED.store(true, Ordering::SeqCst);
        if THREAD_ALIVE.load(Ordering::SeqCst) {
            return;
        }
        THREAD_ALIVE.store(true, Ordering::SeqCst);
        let spawned = std::thread::Builder::new()
            .name("handy-input-watch".into())
            .spawn(hook_thread);
        if let Err(e) = spawned {
            warn!("Failed to start input watch thread: {e}");
            THREAD_ALIVE.store(false, Ordering::SeqCst);
            DESIRED.store(false, Ordering::SeqCst);
        }
    }

    /// Remove the hooks (when Swap last is no longer reachable). Cheap when
    /// the watch is not running.
    pub fn stop_input_watch() {
        if !DESIRED.load(Ordering::SeqCst) && !THREAD_ALIVE.load(Ordering::SeqCst) {
            return;
        }
        let _guard = CONTROL.lock().unwrap_or_else(|e| e.into_inner());
        DESIRED.store(false, Ordering::SeqCst);
        let tid = THREAD_ID.load(Ordering::SeqCst);
        if tid != 0 {
            unsafe {
                if let Err(e) = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0)) {
                    warn!("Input watch: failed to stop hook thread: {e}");
                }
            }
        }
    }

    /// Send the probe and wait for the hook to see it.
    fn probe_alive() -> bool {
        let base = base();
        let sent = now_us(base) + 1;
        let input = INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(UNASSIGNED_VK as u16),
                    wScan: 0,
                    dwFlags: KEYEVENTF_KEYUP,
                    time: 0,
                    dwExtraInfo: PROBE_MARKER,
                },
            },
        };
        let inserted = unsafe { SendInput(&[input], std::mem::size_of::<INPUT>() as i32) };
        if inserted != 1 {
            debug!("Input watch: probe could not be sent");
            return false;
        }
        let start = Instant::now();
        while start.elapsed() < PROBE_TIMEOUT {
            if PROBE_SEEN.load(Ordering::Acquire) >= sent {
                return true;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        warn!("Input watch: hook did not see the probe (removed by Windows?); re-installing");
        restart_watch();
        false
    }

    /// The hooks stopped working (Windows removes a hook that exceeds
    /// `LowLevelHooksTimeout` without telling us): stop trusting the watch
    /// and have the hook thread re-install them. Input before the re-install
    /// is unknown, so pastes made until then copy instead of swapping; the
    /// next paste after it can be swapped in place again.
    fn restart_watch() {
        let _guard = CONTROL.lock().unwrap_or_else(|e| e.into_inner());
        WATCH_SINCE.store(0, Ordering::SeqCst);
        if !DESIRED.load(Ordering::SeqCst) {
            return;
        }
        let tid = THREAD_ID.load(Ordering::SeqCst);
        if tid != 0 {
            // DESIRED stays true, so the hook thread installs fresh hooks
            // (and a new WATCH_SINCE) right after leaving its message loop.
            unsafe {
                if let Err(e) = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0)) {
                    warn!("Input watch: failed to restart hook thread: {e}");
                }
            }
        }
    }

    /// The watch has run continuously since before `at` and (with `probe`)
    /// its hook is still alive right now, verified with a probe event.
    pub fn watch_verified_since(at: Instant, probe: bool) -> bool {
        let since = WATCH_SINCE.load(Ordering::SeqCst);
        if since == 0 {
            return false;
        }
        let at_us = at.saturating_duration_since(base()).as_micros() as u64 + 1;
        since <= at_us && (!probe || probe_alive()) && WATCH_SINCE.load(Ordering::SeqCst) == since
    }

    /// Snapshot of the recorded presses and their clock origin.
    pub fn recorded_inputs() -> Option<(Vec<(u64, InputCode)>, Instant)> {
        let base = *BASE.get()?;
        let events = RING
            .iter()
            .map(|slot| slot.load(Ordering::Acquire))
            .filter(|&packed| packed != 0)
            .map(|packed| ((packed >> 16) - 1, (packed & 0xFFFF) as InputCode))
            .collect();
        Some((events, base))
    }

    /// When `code` was last pressed (per the hook), if it was recorded.
    pub fn last_press_of(code: InputCode) -> Option<Instant> {
        let (events, base) = recorded_inputs()?;
        events
            .iter()
            .filter(|(_, c)| *c == code)
            .map(|(t, _)| *t)
            .max()
            .map(|t| base + Duration::from_micros(t))
    }

    pub fn foreground_app() -> Option<ForegroundApp> {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.0.is_null() {
                return None;
            }
            let mut pid = 0u32;
            if GetWindowThreadProcessId(hwnd, Some(&mut pid)) == 0 {
                pid = 0;
            }
            Some(ForegroundApp {
                window: hwnd.0 as isize,
                process_path: process_path(pid),
            })
        }
    }

    /// Title of a window from [`foreground_app`]; `None` when it has none.
    pub fn window_title(window: isize) -> Option<String> {
        if window == 0 {
            return None;
        }
        let hwnd = HWND(window as *mut core::ffi::c_void);
        unsafe {
            // GetWindowText on one of Handy's own windows sends WM_GETTEXT to
            // our (possibly busy) main thread and can stall; skip them.
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid == 0 || pid == GetCurrentProcessId() {
                return None;
            }
            let len = GetWindowTextLengthW(hwnd);
            if len <= 0 {
                return None;
            }
            // Titles can change between the two calls; the buffer caps the
            // copy (GetWindowTextW truncates and NUL-terminates).
            let len = (len as usize).min(32_767);
            let mut buffer = vec![0u16; len + 1];
            let copied = GetWindowTextW(hwnd, &mut buffer);
            if copied <= 0 {
                return None;
            }
            let copied = (copied as usize).min(len);
            let title = String::from_utf16_lossy(&buffer[..copied]);
            (!title.trim().is_empty()).then_some(title)
        }
    }

    /// Closes a process handle on every path.
    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    unsafe fn process_path(pid: u32) -> Option<String> {
        if pid == 0 {
            return None;
        }
        let handle = OwnedHandle(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?);
        if handle.0.is_invalid() {
            return None;
        }
        // Long-path aware: retry once with the maximum path length.
        for capacity in [1_024usize, 32_768] {
            let mut buffer = vec![0u16; capacity];
            let mut len = buffer.len() as u32;
            if QueryFullProcessImageNameW(
                handle.0,
                PROCESS_NAME_WIN32,
                PWSTR(buffer.as_mut_ptr()),
                &mut len,
            )
            .is_ok()
            {
                let len = (len as usize).min(buffer.len());
                return Some(String::from_utf16_lossy(&buffer[..len]));
            }
        }
        None
    }

    /// The system clipboard's change counter.
    pub fn clipboard_sequence() -> Option<u32> {
        let seq = unsafe { windows::Win32::System::DataExchange::GetClipboardSequenceNumber() };
        (seq != 0).then_some(seq)
    }

    /// The user's clipboard, saved with every format the reliable paste can
    /// restore (text, HTML, RTF, files, images, ...).
    pub struct ClipboardBackup(crate::paste_tx::ClipboardSnapshot);

    impl ClipboardBackup {
        /// Settles a reliable paste still waiting to restore the clipboard
        /// first: until then the clipboard holds Handy's own transcript, and
        /// borrowing it would defeat that guarded restore.
        pub fn capture() -> Result<Self, String> {
            crate::paste_tx::finish_pending();
            crate::paste_tx::ClipboardSnapshot::capture().map(Self)
        }

        pub fn restore(self) -> Result<(), String> {
            self.0.restore()
        }
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
    use super::{ForegroundApp, InputCode};
    use std::time::{Duration, Instant};

    pub fn start_input_watch() {}

    pub fn stop_input_watch() {}

    pub fn watch_verified_since(_at: Instant, _probe: bool) -> bool {
        false
    }

    pub fn recorded_inputs() -> Option<(Vec<(u64, InputCode)>, Instant)> {
        None
    }

    pub fn last_press_of(_code: InputCode) -> Option<Instant> {
        None
    }

    pub fn foreground_app() -> Option<ForegroundApp> {
        None
    }

    pub fn clipboard_sequence() -> Option<u32> {
        None
    }

    pub fn window_title(_window: isize) -> Option<String> {
        None
    }

    pub fn modifiers_released(_timeout: Duration) -> bool {
        true
    }

    /// In-place swaps never run here (no foreground or input probes).
    pub struct ClipboardBackup;

    impl ClipboardBackup {
        pub fn capture() -> Result<Self, String> {
            Err("clipboard backup is not supported on this platform".to_string())
        }

        pub fn restore(self) -> Result<(), String> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLACK: u64 = 50_000;
    const KEY_A: InputCode = 0x41;
    const KEY_F9: InputCode = 0x78;

    fn check(events: &[(u64, InputCode)], from: u64, trigger: u64, until: u64) -> bool {
        real_input_in(events, INPUT_RING, from, trigger, until, SLACK)
    }

    #[test]
    fn no_presses_means_no_input() {
        assert!(!check(&[], 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn presses_before_the_paste_do_not_count() {
        assert!(!check(&[(500, KEY_A)], 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn typing_after_the_paste_counts() {
        assert!(check(&[(1_500_000, KEY_A)], 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn the_trigger_key_itself_is_excused() {
        // F9 is the swap shortcut, timestamped by the hook 3 ms before Handy.
        let events = [(1_997_000, KEY_F9), (2_040_000, KEY_F9)];
        assert!(!check(&events, 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn other_keys_inside_the_slack_still_count() {
        let events = [(1_980_000, KEY_A), (1_997_000, KEY_F9)];
        assert!(check(&events, 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn same_key_well_before_the_trigger_counts() {
        // The user pressed the trigger's mouse button to click somewhere
        // after the paste, long before the gesture.
        let events = [(1_200_000, MOUSE_LEFT), (2_000_000, MOUSE_LEFT)];
        assert!(check(&events, 1_000, 2_000_000, 2_100_000));
    }

    #[test]
    fn input_between_trigger_and_action_counts() {
        let events = [(2_000_000, KEY_F9), (2_060_000, KEY_A)];
        assert!(check(&events, 1_000, 2_000_000, 2_100_000));
        // ...but not after `until`.
        assert!(!check(&events, 1_000, 2_000_000, 2_050_000));
    }

    #[test]
    fn trigger_before_paste_excuses_nothing_after_it() {
        // Queued trigger at 0.5 s, paste at 1 s, typing at 1.2 s.
        let events = [(500_000, KEY_F9), (1_200_000, KEY_F9)];
        assert!(check(&events, 1_000_000, 500_000, 1_500_000));
    }

    #[test]
    fn handy_injections_and_the_menu_mask_are_not_user_input() {
        // Typing, and other software's injections (enigo's shared default
        // marker included), count.
        assert!(is_user_key_press(0x41, 0));
        assert!(is_user_key_press(0x41, 100));
        assert!(is_user_mouse_press(0));
        // Handy's own enigo events never count.
        assert!(!is_user_key_press(0x41, crate::input::INJECTION_MARKER));
        assert!(!is_user_mouse_press(crate::input::INJECTION_MARKER));
        // handy-keys' menu mask on Win/Alt hotkeys: neither its marker nor
        // the unassigned key itself (whatever the marker) counts.
        assert!(!is_user_key_press(UNASSIGNED_VK, HANDY_KEYS_MARKER));
        assert!(!is_user_key_press(UNASSIGNED_VK, 0));
        assert!(!is_user_key_press(0x41, HANDY_KEYS_MARKER));
        // Modifiers alone never count.
        for &vk in MODIFIER_VKS {
            assert!(!is_user_key_press(vk, 0), "{vk:#x}");
        }
        assert_ne!(crate::input::INJECTION_MARKER, enigo::EVENT_MARKER as usize);
    }

    #[test]
    fn a_ring_full_of_newer_presses_counts_as_input() {
        let events: Vec<_> = (0..INPUT_RING as u64)
            .map(|i| (2_000_000 + i * 10, KEY_F9))
            .collect();
        assert!(check(&events, 1_000, 2_000_000, 3_000_000));
    }
}
