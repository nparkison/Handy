//! "One-button cockpit": last-dictation actions driven from one button.
//!
//! - **Paste last** pastes the newest dictation again (cleaned-up first).
//! - **Swap last** replaces the text Handy just pasted with the other version
//!   (original <-> cleaned-up). It only edits in place when every guard in
//!   [`swap::plan_swap`] passes; otherwise it copies the other version.
//! - Late cleanup (a deadline miss, see [`deadline`]) lands in History and can
//!   offer "Cleanup ready · Swap".
//! - Tap gestures on the main binding trigger both (see [`gestures`]).
//!
//! ## In-place swap mechanism and its limits
//! Handy remembers its last paste: what was inserted (including a trailing
//! space), when, and into which foreground window. To swap it selects the
//! inserted text back with Shift+Left × N characters and pastes the other
//! version over the selection through the normal paste path. Never Ctrl+Z,
//! never backspace runs. Limits: apps that rewrite pasted text (autocorrect,
//! smart quotes, auto-indent, rich-text conversion) or move the caret by
//! something other than one character per Left press make N wrong, which the
//! guards can only reduce (single-line, plain characters only, same window,
//! no key/mouse-button input since the paste, < 2 min, not a terminal, no
//! auto-submit). Input after the swap trigger itself is not detected. Only
//! Windows can verify the window and input guards; other platforms always copy.

pub mod deadline;
pub mod gestures;
mod platform;
pub mod swap;

use crate::actions::ShortcutAction;
use crate::input::EnigoState;
use crate::managers::history::{CleanupState, HistoryEntry, HistoryManager};
use crate::overlay_notice::{show_overlay_notice, Notice, NoticeAction, NoticeText};
use crate::settings::{get_settings, AppSettings, PasteMethod};
use crate::TranscriptionCoordinator;
use log::{debug, error, info, warn};
use platform::ForegroundApp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use swap::{plan_swap, SwapCheck, SwapPlan};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_clipboard_manager::ClipboardExt;

/// How long Paste last's confirmation stays up.
const PASTED_LAST_NOTICE: Duration = Duration::from_millis(1_000);
/// How long to wait for the user to let go of Ctrl/Alt/Shift/Win before
/// injecting keys.
const MODIFIER_RELEASE_TIMEOUT: Duration = Duration::from_millis(1_000);
/// A click on the overlay's Swap button: the mouse-down happened a moment
/// before the action runs; inputs older than this still count as "since paste".
const NOTICE_CLICK_ALLOWANCE: Duration = Duration::from_millis(600);

/// Which version of a dictation was pasted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    Original,
    CleanedUp,
}

/// The cleaned-up version as far as the last paste knows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Polished {
    None,
    /// Missed the deadline; the request is still running.
    Pending,
    Ready(String),
}

impl Polished {
    fn from_entry(entry: &HistoryEntry) -> Self {
        if entry.cleanup_state == Some(CleanupState::Pending) {
            return Polished::Pending;
        }
        match &entry.post_processed_text {
            Some(text) if !text.trim().is_empty() => Polished::Ready(text.clone()),
            _ => Polished::None,
        }
    }
}

/// Handy's most recent paste into another app.
#[derive(Clone, Debug)]
struct LastPaste {
    entry_id: Option<i64>,
    original: String,
    polished: Polished,
    pasted: Version,
    /// Exactly what was inserted (trailing space included).
    inserted_text: String,
    at: Instant,
    foreground: Option<ForegroundApp>,
}

static LAST_PASTE: Mutex<Option<LastPaste>> = Mutex::new(None);
/// Bumped whenever a recording turns out to be a real dictation.
static DICTATIONS: AtomicU64 = AtomicU64::new(0);

fn last_paste() -> std::sync::MutexGuard<'static, Option<LastPaste>> {
    LAST_PASTE.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn note_dictation_started() -> u64 {
    DICTATIONS.fetch_add(1, Ordering::AcqRel) + 1
}

pub fn dictation_count() -> u64 {
    DICTATIONS.load(Ordering::Acquire)
}

fn notice(app: &AppHandle, key: &str) {
    show_overlay_notice(app, Notice::info(NoticeText::new(key)));
}

fn pipeline_busy(app: &AppHandle) -> bool {
    app.try_state::<TranscriptionCoordinator>()
        .is_some_and(|c| c.is_busy())
}

/// What `clipboard::paste` actually inserts for `text`.
fn inserted_text(settings: &AppSettings, text: &str) -> String {
    if settings.append_trailing_space {
        format!("{text} ")
    } else {
        text.to_string()
    }
}

/// Remember a successful Handy paste so Swap last can act on it. Call right
/// after the paste returns, on the thread that pasted.
pub fn record_paste(
    app: &AppHandle,
    entry_id: Option<i64>,
    original: String,
    polished: Polished,
    pasted: Version,
    pasted_text: &str,
) {
    let settings = get_settings(app);
    platform::start_input_watch();
    *last_paste() = Some(LastPaste {
        entry_id,
        original,
        polished,
        pasted,
        inserted_text: inserted_text(&settings, pasted_text),
        at: Instant::now(),
        foreground: platform::foreground_app(),
    });
}

/// A deadline-missed cleanup resolved: keep the last paste in sync so Swap
/// last can use (or report the absence of) the cleaned-up version.
fn on_late_cleanup_resolved(entry_id: i64, cleaned: Option<&str>) {
    if let Some(last) = last_paste().as_mut() {
        if last.entry_id == Some(entry_id) && last.polished == Polished::Pending {
            last.polished = match cleaned {
                Some(text) if !text.trim().is_empty() => Polished::Ready(text.to_string()),
                _ => Polished::None,
            };
        }
    }
}

/// Finish a deadline-missed cleanup in the background. The result only ever
/// goes to the History entry (never auto-replaces pasted text); when it is
/// still fresh it offers "Cleanup ready · Swap".
pub fn spawn_late_cleanup<F>(app: AppHandle, task: F, entry_id: Option<i64>, prompt: Option<String>)
where
    F: std::future::Future<Output = Option<String>> + Send + 'static,
{
    let missed_at = Instant::now();
    let dictations_at_miss = dictation_count();
    tauri::async_runtime::spawn(async move {
        let cleaned = tokio::time::timeout(deadline::BACKGROUND_CAP, task)
            .await
            .unwrap_or_else(|_| {
                info!(
                    "Late cleanup abandoned after {:?}",
                    deadline::BACKGROUND_CAP
                );
                None
            });
        info!(
            "Late cleanup {} after {:?}",
            if cleaned.is_some() {
                "arrived"
            } else {
                "failed"
            },
            missed_at.elapsed()
        );
        let Some(id) = entry_id else {
            return;
        };
        let hm = app.state::<Arc<HistoryManager>>();
        let resolved =
            hm.resolve_late_cleanup(id, cleaned.clone().map(|text| (text, prompt.clone())));
        match resolved {
            Ok(Some(_)) => {}
            Ok(None) => {
                debug!("History entry {id} no longer awaits cleanup");
                return;
            }
            Err(e) => {
                error!("Failed to save late cleanup for entry {id}: {e}");
                return;
            }
        }
        on_late_cleanup_resolved(id, cleaned.as_deref());
        if cleaned.is_none() {
            return;
        }

        let settings = get_settings(&app);
        let offer = deadline::should_offer_late_swap(
            missed_at.elapsed(),
            dictation_count() != dictations_at_miss,
            pipeline_busy(&app),
            settings.swap_last_reachable(),
        );
        // Swap only makes sense while that entry is still the last paste.
        let is_last = last_paste()
            .as_ref()
            .is_some_and(|last| last.entry_id == Some(id) && last.pasted == Version::Original);
        if offer && is_last {
            let key = if settings.tap_gestures_active() {
                "overlay.notice.cleanupReadyDoubleTap"
            } else {
                "overlay.notice.cleanupReady"
            };
            show_overlay_notice(
                &app,
                Notice::warning(NoticeText::new(key)).with_action(NoticeAction::new(
                    NoticeText::new("overlay.notice.swap"),
                    |app| {
                        let clicked_at = Instant::now()
                            .checked_sub(NOTICE_CLICK_ALLOWANCE)
                            .unwrap_or_else(Instant::now);
                        swap_last(app, clicked_at);
                    },
                )),
            );
        }
    });
}

/// Run `f` on the main thread (where the regular pipeline pastes) and wait.
fn on_main_thread<T: Send + 'static>(
    app: &AppHandle,
    f: impl FnOnce(&AppHandle) -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = app.clone();
    if let Err(e) = app.run_on_main_thread(move || {
        let _ = tx.send(f(&handle));
    }) {
        error!("Failed to run on main thread: {e:?}");
        return None;
    }
    rx.recv().ok()
}

/// Paste the newest dictation again (cleaned-up version if available). Never
/// creates a History entry. Blocks; call from a background thread.
pub fn paste_last(app: &AppHandle) {
    let hm = app.state::<Arc<HistoryManager>>();
    let entry = match hm.get_latest_completed_entry() {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            notice(app, "overlay.notice.nothingToPaste");
            return;
        }
        Err(e) => {
            error!("Paste last: failed to read history: {e}");
            return;
        }
    };
    let polished = Polished::from_entry(&entry);
    let (version, text) = match &polished {
        Polished::Ready(text) => (Version::CleanedUp, text.clone()),
        // Still in cleanup (or none): paste the original, like the deadline.
        _ => (Version::Original, entry.transcription_text.clone()),
    };
    if !platform::modifiers_released(MODIFIER_RELEASE_TIMEOUT) {
        warn!("Paste last: modifier keys still held; pasting anyway");
    }

    let original = entry.transcription_text.clone();
    let entry_id = entry.id;
    let result = on_main_thread(app, move |app| {
        let result = crate::utils::paste(text.clone(), app.clone());
        if result.is_ok() {
            record_paste(app, Some(entry_id), original, polished, version, &text);
        }
        result
    });
    match result {
        Some(Ok(())) => {
            show_overlay_notice(
                app,
                Notice::info(NoticeText::new("overlay.notice.pastedLast"))
                    .with_duration(PASTED_LAST_NOTICE),
            );
        }
        Some(Err(e)) => {
            error!("Paste last failed: {e}");
            let _ = app.emit("paste-error", ());
        }
        None => {}
    }
}

/// The version to swap to, or the notice explaining why there is none.
fn swap_target(last: &LastPaste) -> Result<(Version, String), &'static str> {
    match last.pasted {
        Version::CleanedUp => Ok((Version::Original, last.original.clone())),
        Version::Original => match &last.polished {
            Polished::Ready(text) if text.trim() != last.original.trim() => {
                Ok((Version::CleanedUp, text.clone()))
            }
            Polished::Pending => Err("overlay.notice.cleanupStillRunning"),
            _ => Err("overlay.notice.noCleanedUpVersion"),
        },
    }
}

fn select_back(app: &AppHandle, chars: usize) -> Result<(), String> {
    use enigo::{Direction, Key, Keyboard};
    let state = app
        .try_state::<EnigoState>()
        .ok_or("Enigo state not initialized")?;
    let mut enigo = state
        .0
        .lock()
        .map_err(|e| format!("Failed to lock Enigo: {e}"))?;
    enigo
        .key(Key::Shift, Direction::Press)
        .map_err(|e| format!("Failed to press Shift: {e}"))?;
    let mut result = Ok(());
    for _ in 0..chars {
        if let Err(e) = enigo.key(Key::LeftArrow, Direction::Click) {
            result = Err(format!("Failed to press Left: {e}"));
            break;
        }
    }
    // Always release Shift, even after a failure.
    enigo
        .key(Key::Shift, Direction::Release)
        .map_err(|e| format!("Failed to release Shift: {e}"))?;
    result
}

/// Swap the last Handy paste to its other version. `trigger_at` is when the
/// user started the swap gesture/shortcut: real input between the paste and
/// it means the caret may have moved, so the swap copies instead.
/// Blocks; call from a background thread.
pub fn swap_last(app: &AppHandle, trigger_at: Instant) {
    let Some(last) = last_paste().clone() else {
        notice(app, "overlay.notice.nothingToSwap");
        return;
    };
    let (target, text) = match swap_target(&last) {
        Ok(target) => target,
        Err(key) => {
            notice(app, key);
            return;
        }
    };

    let settings = get_settings(app);
    let modifiers_released = platform::modifiers_released(MODIFIER_RELEASE_TIMEOUT);
    let foreground = platform::foreground_app();
    let same_window = match (&last.foreground, &foreground) {
        (Some(then), Some(now)) => Some(then.window == now.window),
        _ => None,
    };
    let paste_method_ok = matches!(
        settings.paste_method,
        PasteMethod::CtrlV
            | PasteMethod::CtrlShiftV
            | PasteMethod::ShiftInsert
            | PasteMethod::Direct
    ) && !settings.auto_submit;
    let check = SwapCheck {
        inserted_text: &last.inserted_text,
        age: last.at.elapsed(),
        same_window,
        input_since_paste: platform::input_since_paste(last.at, trigger_at),
        process_name: last
            .foreground
            .as_ref()
            .and_then(|f| f.process_path.as_deref()),
        paste_method_ok,
        modifiers_released,
    };
    let plan = plan_swap(&check);
    debug!("Swap last: {plan:?}");

    let select_back_chars = match plan {
        SwapPlan::InPlace { select_back } => select_back,
        SwapPlan::CopyOnly(reason) => {
            info!("Swap last: copying instead of swapping in place ({reason:?})");
            match app.clipboard().write_text(text) {
                Ok(()) => notice(app, "overlay.notice.swapCopied"),
                Err(e) => error!("Swap last: failed to copy: {e}"),
            }
            return;
        }
    };

    let result = on_main_thread(app, move |app| {
        select_back(app, select_back_chars)?;
        crate::utils::paste(text.clone(), app.clone())?;
        Ok::<String, String>(text)
    });
    match result {
        Some(Ok(text)) => {
            {
                let settings = get_settings(app);
                let mut guard = last_paste();
                if let Some(record) = guard.as_mut() {
                    record.pasted = target;
                    record.inserted_text = inserted_text(&settings, &text);
                    record.at = Instant::now();
                    record.foreground = foreground;
                }
            }
            notice(
                app,
                match target {
                    Version::CleanedUp => "overlay.notice.swappedToCleanedUp",
                    Version::Original => "overlay.notice.swappedToOriginal",
                },
            );
        }
        Some(Err(e)) => {
            error!("Swap last failed: {e}");
            // The selection state is unknown now; never retry in place.
            last_paste().take();
            let _ = app.emit("paste-error", ());
        }
        None => {}
    }
}

/* ───────────────────────────── shortcut actions ───────────────────────── */

/// When the Swap last / Paste last key went down.
static ACTION_PRESSED_AT: Mutex<Option<Instant>> = Mutex::new(None);

fn run_action(app: &AppHandle, name: &str, f: impl FnOnce(&AppHandle, Instant) + Send + 'static) {
    let pressed_at = ACTION_PRESSED_AT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_else(Instant::now);
    if pipeline_busy(app) {
        debug!("{name}: ignored while a dictation is in progress");
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || f(&app, pressed_at));
}

/// Bindable "Paste last". Acts on key release so held shortcut keys do not
/// mix with the injected paste chord.
pub struct PasteLastAction;

impl ShortcutAction for PasteLastAction {
    fn start(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        *ACTION_PRESSED_AT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    fn stop(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        run_action(app, "Paste last", |app, _| paste_last(app));
    }
}

/// Bindable "Swap last". Acts on key release; the press time bounds the
/// "no input since the paste" guard.
pub struct SwapLastAction;

impl ShortcutAction for SwapLastAction {
    fn start(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        *ACTION_PRESSED_AT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    fn stop(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        run_action(app, "Swap last", swap_last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last(pasted: Version, polished: Polished) -> LastPaste {
        LastPaste {
            entry_id: Some(1),
            original: "um hello".into(),
            polished,
            pasted,
            inserted_text: "um hello".into(),
            at: Instant::now(),
            foreground: None,
        }
    }

    #[test]
    fn swap_target_flips_between_versions() {
        let ready = Polished::Ready("Hello.".into());
        assert_eq!(
            swap_target(&last(Version::Original, ready.clone())),
            Ok((Version::CleanedUp, "Hello.".to_string()))
        );
        assert_eq!(
            swap_target(&last(Version::CleanedUp, ready)),
            Ok((Version::Original, "um hello".to_string()))
        );
    }

    #[test]
    fn swap_target_explains_missing_versions() {
        assert_eq!(
            swap_target(&last(Version::Original, Polished::Pending)),
            Err("overlay.notice.cleanupStillRunning")
        );
        assert_eq!(
            swap_target(&last(Version::Original, Polished::None)),
            Err("overlay.notice.noCleanedUpVersion")
        );
        // Identical cleanup: nothing different to swap to.
        assert_eq!(
            swap_target(&last(
                Version::Original,
                Polished::Ready(" um hello ".into())
            )),
            Err("overlay.notice.noCleanedUpVersion")
        );
    }
}
