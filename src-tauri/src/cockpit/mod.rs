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
//! inserted text back with Shift+Left × N characters, copies the selection
//! (Ctrl+Insert) to verify it is exactly what Handy inserted, and only then
//! pastes the other version over it through the normal paste path. Never
//! Ctrl+Z, never backspace runs. Guards (all must pass, otherwise the other
//! version is only copied): clipboard paste method without auto-submit,
//! single-line plain characters only, same window, a known non-terminal app,
//! no key/mouse-button input (other than the trigger's own) from the paste
//! until the swap acts, a live input watch, < 2 min. Only Windows can verify
//! the window and input guards; other platforms always copy.

pub mod deadline;
pub mod gestures;
pub(crate) mod platform;
pub mod swap;

use crate::actions::ShortcutAction;
use crate::input::EnigoState;
use crate::managers::history::{CleanupState, HistoryEntry, HistoryManager, LateCleanup};
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
/// A click on the overlay's Swap button: when the hook did not record the
/// button-down, assume it happened this long before the action runs.
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
    /// Increases with every recorded paste (and swap): a swap only acts on,
    /// and only updates, the paste it planned against.
    id: u64,
    /// Identifies a deadline-missed cleanup for this paste, if any.
    late_id: Option<u64>,
    original: String,
    polished: Polished,
    pasted: Version,
    /// Exactly what was inserted (trailing space included).
    inserted_text: String,
    /// Taken right before the paste started (so input during the paste's own
    /// delays counts as "since the paste").
    at: Instant,
    foreground: Option<ForegroundApp>,
    /// The paste method inserted plain text Handy can select back (clipboard
    /// paste chords only) and auto-submit was off, as configured at paste time.
    paste_method_ok: bool,
}

static LAST_PASTE: Mutex<Option<LastPaste>> = Mutex::new(None);
static NEXT_PASTE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_LATE_ID: AtomicU64 = AtomicU64::new(1);
/// The most recent late-cleanup resolution, for a paste recorded after its
/// cleanup already finished (`(late_id, result)`).
static LATE_RESOLVED: Mutex<Option<(u64, Polished)>> = Mutex::new(None);
/// Swaps run one at a time.
static SWAP_LOCK: Mutex<()> = Mutex::new(());
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

/// A fresh id for a deadline-missed cleanup (see [`spawn_late_cleanup`]).
pub fn new_late_id() -> u64 {
    NEXT_LATE_ID.fetch_add(1, Ordering::AcqRel)
}

/// Settings changed: refresh the gesture cache, and run the input watch only
/// while Swap last is reachable (installed before the first paste, so that
/// paste can already be swapped in place). Cheap; called on every settings
/// read and write.
pub fn sync_settings(settings: &AppSettings) {
    gestures::cache_settings(settings);
    if settings.swap_last_reachable() {
        platform::start_input_watch();
    } else {
        platform::stop_input_watch();
    }
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

/// Paste methods whose result Shift+Left can select back reliably. Direct
/// typing is left out: typed text triggers autocorrect, auto-pairing and
/// smart quotes, so what landed may differ from what Handy typed.
fn paste_method_allows_swap(settings: &AppSettings) -> bool {
    matches!(
        settings.paste_method,
        PasteMethod::CtrlV | PasteMethod::CtrlShiftV | PasteMethod::ShiftInsert
    ) && !settings.auto_submit
}

/// Remember a successful Handy paste so Swap last can act on it. Call right
/// after the paste returns, on the thread that pasted; `started_at` is when
/// the paste began.
pub fn record_paste(
    app: &AppHandle,
    late_id: Option<u64>,
    original: String,
    polished: Polished,
    pasted: Version,
    pasted_text: &str,
    started_at: Instant,
) {
    let settings = get_settings(app);
    if settings.swap_last_reachable() {
        platform::start_input_watch();
    }
    // A late cleanup that finished before this paste was recorded.
    let polished = match (late_id, polished) {
        (Some(late), Polished::Pending) => {
            let resolved = LATE_RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
            match resolved.as_ref() {
                Some((id, result)) if *id == late => result.clone(),
                _ => Polished::Pending,
            }
        }
        (_, polished) => polished,
    };
    *last_paste() = Some(LastPaste {
        id: NEXT_PASTE_ID.fetch_add(1, Ordering::AcqRel),
        late_id,
        original,
        polished,
        pasted,
        inserted_text: inserted_text(&settings, pasted_text),
        at: started_at,
        foreground: platform::foreground_app(),
        paste_method_ok: paste_method_allows_swap(&settings),
    });
}

fn polished_from_late(cleaned: Option<&str>) -> Polished {
    match cleaned {
        Some(text) if !text.trim().is_empty() => Polished::Ready(text.to_string()),
        _ => Polished::None,
    }
}

/// A deadline-missed cleanup resolved: keep the last paste in sync so Swap
/// last can use (or report the absence of) the cleaned-up version. Works
/// without a History entry, and for a paste recorded later.
fn on_late_cleanup_resolved(late_id: u64, cleaned: Option<&str>) {
    let result = polished_from_late(cleaned);
    *LATE_RESOLVED.lock().unwrap_or_else(|e| e.into_inner()) = Some((late_id, result.clone()));
    if let Some(last) = last_paste().as_mut() {
        if last.late_id == Some(late_id) && last.polished == Polished::Pending {
            last.polished = result;
        }
    }
}

/// Finish a deadline-missed cleanup in the background. The result only ever
/// goes to the History entry (never auto-replaces pasted text) and to the
/// in-memory last paste; when it is still fresh it offers "Cleanup ready ·
/// Swap". Must be started for every missed cleanup, whatever happens to the
/// paste, so the entry never stays "Cleaning up".
pub fn spawn_late_cleanup<F>(
    app: AppHandle,
    task: F,
    entry_id: Option<i64>,
    late_id: u64,
    prompt: Option<String>,
) where
    F: std::future::Future<Output = Option<(String, bool)>> + Send + 'static,
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
        // In-memory state first: Swap works even when History could not be
        // updated (no entry because the WAV failed to save, DB error, ...).
        on_late_cleanup_resolved(late_id, cleaned.as_ref().map(|(text, _)| text.as_str()));
        if let Some(id) = entry_id {
            let hm = app.state::<Arc<HistoryManager>>();
            let late = cleaned.clone().map(|(text, screenshot)| LateCleanup {
                text,
                prompt: prompt.clone(),
                screenshot,
            });
            match hm.resolve_late_cleanup(id, late) {
                Ok(Some(_)) => {}
                Ok(None) => debug!("History entry {id} no longer awaits cleanup"),
                Err(e) => error!("Failed to save late cleanup for entry {id}: {e}"),
            }
        }
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
        // Swap only makes sense while that dictation is still the last paste.
        let is_last = last_paste()
            .as_ref()
            .is_some_and(|last| last.late_id == Some(late_id) && last.pasted == Version::Original);
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
                    |app| swap_last(app, overlay_click_time()),
                )),
            );
        }
    });
}

/// When the overlay's Swap button was pressed: the hook's record of the left
/// button going down, else a conservative estimate.
fn overlay_click_time() -> Instant {
    platform::last_press_of(platform::MOUSE_LEFT)
        .filter(|at| at.elapsed() <= NOTICE_CLICK_ALLOWANCE * 5)
        .unwrap_or_else(|| {
            Instant::now()
                .checked_sub(NOTICE_CLICK_ALLOWANCE)
                .unwrap_or_else(Instant::now)
        })
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
    // A paste chord mixed with a held Ctrl/Alt/Win becomes another shortcut.
    if !platform::modifiers_released(MODIFIER_RELEASE_TIMEOUT) {
        warn!("Paste last: modifier keys still held; not pasting");
        notice(app, "overlay.notice.releaseKeys");
        return;
    }

    let original = entry.transcription_text.clone();
    let result = on_main_thread(app, move |app| {
        let started_at = Instant::now();
        let result = crate::utils::paste(text.clone(), app.clone());
        if result.is_ok() {
            record_paste(app, None, original, polished, version, &text, started_at);
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

fn with_enigo<T>(
    app: &AppHandle,
    f: impl FnOnce(&mut enigo::Enigo) -> Result<T, String>,
) -> Result<T, String> {
    let state = app
        .try_state::<EnigoState>()
        .ok_or("Enigo state not initialized")?;
    let mut enigo = state
        .0
        .lock()
        .map_err(|e| format!("Failed to lock Enigo: {e}"))?;
    f(&mut enigo)
}

fn select_back(app: &AppHandle, chars: usize) -> Result<(), String> {
    use enigo::{Direction, Key, Keyboard};
    with_enigo(app, |enigo| {
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
    })
}

/// Collapse the selection to its right end (where the caret was).
fn collapse_selection(app: &AppHandle) {
    use enigo::{Direction, Key, Keyboard};
    if let Err(e) = with_enigo(app, |enigo| {
        enigo
            .key(Key::RightArrow, Direction::Click)
            .map_err(|e| format!("Failed to press Right: {e}"))
    }) {
        warn!("Swap last: {e}");
    }
}

/// How long to wait for the target app to put the selection on the clipboard.
const COPY_SELECTION_TIMEOUT: Duration = Duration::from_millis(400);

/// Copy the current selection with Ctrl+Insert (never Ctrl+C, which is
/// SIGINT in terminals) and return it, restoring the user's clipboard.
/// `None` when the app did not copy anything.
fn copy_selection(app: &AppHandle) -> Option<String> {
    use enigo::{Direction, Key, Keyboard};
    let before = platform::clipboard_sequence()?;
    let clipboard = app.clipboard();
    let saved_text = clipboard.read_text().ok().filter(|t| !t.is_empty());
    let saved_image = if saved_text.is_none() {
        clipboard.read_image().ok().map(|image| image.to_owned())
    } else {
        None
    };
    let sent = with_enigo(app, |enigo| {
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| format!("Failed to press Ctrl: {e}"))?;
        let result = enigo
            .key(Key::Insert, Direction::Click)
            .map_err(|e| format!("Failed to press Insert: {e}"));
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| format!("Failed to release Ctrl: {e}"))?;
        result
    });
    if let Err(e) = sent {
        warn!("Swap last: {e}");
        return None;
    }
    let start = Instant::now();
    let mut changed = false;
    while start.elapsed() < COPY_SELECTION_TIMEOUT {
        if platform::clipboard_sequence().is_some_and(|now| now != before) {
            changed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let copied = if changed {
        // Let the owner finish writing all formats.
        std::thread::sleep(Duration::from_millis(20));
        clipboard.read_text().ok()
    } else {
        None
    };
    if changed {
        if let Some(text) = saved_text {
            let _ = clipboard.write_text(text);
        } else if let Some(image) = saved_image {
            let _ = clipboard.write_image(&image);
        } else {
            let _ = clipboard.clear();
        }
    }
    copied
}

/// Does the copied selection equal what Handy inserted? Editors may store a
/// typed trailing space as a no-break space.
fn selection_matches(copied: &str, inserted: &str) -> bool {
    let normalize = |s: &str| s.replace('\u{A0}', " ");
    normalize(copied) == normalize(inserted)
}

enum SwapResult {
    Swapped,
    /// A guard failed at the last moment; nothing was changed.
    CopyInstead(&'static str),
    /// A newer paste replaced the one this swap planned against.
    Stale,
    Failed(String),
}

/// Swap the last Handy paste to its other version. `trigger_at` is when the
/// user started the swap gesture/shortcut: real input between the paste and
/// the moment the swap acts means the caret may have moved, so the swap
/// copies instead. Blocks; call from a background thread.
pub fn swap_last(app: &AppHandle, trigger_at: Instant) {
    let _serialized = SWAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    let modifiers_released = platform::modifiers_released(MODIFIER_RELEASE_TIMEOUT);
    let foreground = platform::foreground_app();
    let same_window = match (&last.foreground, &foreground) {
        (Some(then), Some(now)) => Some(then.window == now.window),
        _ => None,
    };
    let check = SwapCheck {
        inserted_text: &last.inserted_text,
        age: last.at.elapsed(),
        same_window,
        input_since_paste: platform::input_since_paste(last.at, trigger_at, Instant::now(), true),
        process_name: last
            .foreground
            .as_ref()
            .and_then(|f| f.process_path.as_deref()),
        paste_method_ok: last.paste_method_ok,
        modifiers_released,
    };
    let plan = plan_swap(&check);
    debug!("Swap last: {plan:?}");

    let copy_instead = |app: &AppHandle, text: String| match app.clipboard().write_text(text) {
        Ok(()) => notice(app, "overlay.notice.swapCopied"),
        Err(e) => error!("Swap last: failed to copy: {e}"),
    };

    let select_back_chars = match plan {
        SwapPlan::InPlace { select_back } => select_back,
        SwapPlan::CopyOnly(reason) => {
            info!("Swap last: copying instead of swapping in place ({reason:?})");
            copy_instead(app, text);
            return;
        }
    };

    let planned = last.clone();
    let window = foreground.as_ref().map(|f| f.window);
    let swap_text = text.clone();
    let result = on_main_thread(app, move |app| {
        // Re-validate right before touching the target app: a paste queued
        // on this thread may have landed since the plan was made.
        if last_paste().as_ref().map(|l| l.id) != Some(planned.id) {
            return SwapResult::Stale;
        }
        if platform::input_since_paste(planned.at, trigger_at, Instant::now(), false) != Some(false)
        {
            return SwapResult::CopyInstead("input after the trigger");
        }
        if platform::foreground_app().map(|f| f.window) != window {
            return SwapResult::CopyInstead("focus changed");
        }
        if let Err(e) = select_back(app, select_back_chars) {
            return SwapResult::Failed(e);
        }
        // Verify the selection is exactly what Handy inserted before
        // replacing it: apps can rewrite pasted text, and a stale clipboard
        // restore can paste something else.
        match copy_selection(app) {
            Some(copied) if selection_matches(&copied, &planned.inserted_text) => {}
            other => {
                debug!(
                    "Swap last: selection {} the inserted text",
                    if other.is_some() {
                        "differs from"
                    } else {
                        "could not be compared with"
                    }
                );
                collapse_selection(app);
                return SwapResult::CopyInstead("selection mismatch");
            }
        }
        let started_at = Instant::now();
        match crate::utils::paste(swap_text.clone(), app.clone()) {
            Ok(()) => {
                let settings = get_settings(app);
                let mut guard = last_paste();
                if let Some(record) = guard.as_mut().filter(|r| r.id == planned.id) {
                    record.id = NEXT_PASTE_ID.fetch_add(1, Ordering::AcqRel);
                    record.pasted = target;
                    record.inserted_text = inserted_text(&settings, &swap_text);
                    record.at = started_at;
                    record.foreground = platform::foreground_app();
                }
                SwapResult::Swapped
            }
            Err(e) => SwapResult::Failed(e),
        }
    });
    match result {
        Some(SwapResult::Swapped) => notice(
            app,
            match target {
                Version::CleanedUp => "overlay.notice.swappedToCleanedUp",
                Version::Original => "overlay.notice.swappedToOriginal",
            },
        ),
        Some(SwapResult::CopyInstead(reason)) => {
            info!("Swap last: copying instead of swapping in place ({reason})");
            copy_instead(app, text);
        }
        Some(SwapResult::Stale) => {
            debug!("Swap last: a newer paste landed; not swapping");
            notice(app, "overlay.notice.nothingToSwap");
        }
        Some(SwapResult::Failed(e)) => {
            error!("Swap last failed: {e}");
            // The selection state is unknown now; never retry in place.
            let mut guard = last_paste();
            if guard.as_ref().is_some_and(|r| r.id == last.id) {
                guard.take();
            }
            drop(guard);
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
            id: 1,
            late_id: None,
            original: "um hello".into(),
            polished,
            pasted,
            inserted_text: "um hello".into(),
            at: Instant::now(),
            foreground: None,
            paste_method_ok: true,
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
    fn selection_check_is_exact_except_no_break_space() {
        assert!(selection_matches("Hello. ", "Hello. "));
        assert!(selection_matches("Hello.\u{A0}", "Hello. "));
        assert!(!selection_matches("Hello.", "Hello. "));
        assert!(!selection_matches("Helo. ", "Hello. "));
    }

    #[test]
    fn late_result_reaches_a_paste_recorded_afterwards() {
        let late = new_late_id();
        on_late_cleanup_resolved(late, Some("Cleaned."));
        let resolved = LATE_RESOLVED.lock().unwrap().clone();
        assert_eq!(resolved, Some((late, Polished::Ready("Cleaned.".into()))));
        // A blank result means there is no cleaned-up version.
        on_late_cleanup_resolved(late, Some("  "));
        let resolved = LATE_RESOLVED.lock().unwrap().clone();
        assert_eq!(resolved, Some((late, Polished::None)));
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
