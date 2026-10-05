//! Transient notices in the recording overlay.
//!
//! The main window is usually hidden, so in-app toasts are never seen. A
//! notice is a short state of the recording overlay itself: an icon, one line
//! of text, and an optional action button. Only one notice shows at a time (a
//! newer one replaces it), and any new dictation press clears it immediately.
//!
//! # Usage
//!
//! ```ignore
//! use crate::overlay_notice::{show_overlay_notice, Notice, NoticeAction, NoticeText};
//!
//! // Info: ~1.5 s, dropped when the overlay is off.
//! show_overlay_notice(&app, Notice::info(NoticeText::new("overlay.notice.switchedMic").param("mic", &name)));
//!
//! // Warning with an action: ~4 s (paused while hovered). The closure runs on a
//! // background thread when the button is clicked; it may show a follow-up
//! // notice, otherwise the overlay hides once it returns.
//! show_overlay_notice(
//!     &app,
//!     Notice::warning(NoticeText::new("overlay.notice.heardNothing").param("mic", &mic))
//!         .urgent() // aria-live="assertive"
//!         .with_action(NoticeAction::new(
//!             NoticeText::new("overlay.notice.useMic").param("mic", &other),
//!             move |app| switch_microphone(app, &other),
//!         ))
//!         .with_tray_fallback(localized_tooltip_line),
//! );
//! ```
//!
//! Text is an i18n key plus string params, rendered by the overlay webview with
//! i18next (every key must exist in all locales). Param values longer than ~28
//! characters are truncated on screen and shown in full in a tooltip.
//!
//! When Show Overlay is off, info notices are dropped and warnings fall back to
//! a line in the tray tooltip (`with_tray_fallback`, which the caller must pass
//! already localized because Rust only has the tray strings). That line is
//! cleared by the next press.
//!
//! Action buttons are clickable: the overlay window is non-focusable (it never
//! steals focus) but not click-through, the same way its cancel button works.

use crate::settings::{self, OverlayStyle};
use log::debug;
use serde::Serialize;
use specta::Type;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};

/// Default on-screen time of an info notice.
pub const INFO_DURATION: Duration = Duration::from_millis(1_500);
/// Default on-screen time of a warning (paused while the pointer hovers).
pub const WARNING_DURATION: Duration = Duration::from_millis(4_000);
/// Backend safety net: a notice is force-hidden this long after its duration
/// even if the overlay webview never reports its timer (e.g. it hung). Bounds
/// how long hovering can keep a notice up.
const MAX_HOVER_EXTENSION: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum NoticeKind {
    Info,
    Warning,
}

/// An i18n key plus its interpolation params, rendered by the overlay.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Type)]
pub struct NoticeText {
    pub key: String,
    pub params: BTreeMap<String, String>,
}

impl NoticeText {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            params: BTreeMap::new(),
        }
    }

    pub fn param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }
}

pub type NoticeHandler = Arc<dyn Fn(&AppHandle) + Send + Sync + 'static>;

/// The optional button on a notice.
#[derive(Clone)]
pub struct NoticeAction {
    pub label: NoticeText,
    pub run: NoticeHandler,
}

impl NoticeAction {
    pub fn new(label: NoticeText, run: impl Fn(&AppHandle) + Send + Sync + 'static) -> Self {
        Self {
            label,
            run: Arc::new(run),
        }
    }
}

/// A transient overlay notice. Build with [`Notice::info`] / [`Notice::warning`].
#[derive(Clone)]
pub struct Notice {
    pub kind: NoticeKind,
    pub message: NoticeText,
    pub action: Option<NoticeAction>,
    pub duration: Duration,
    /// Announce with `aria-live="assertive"` instead of `polite`.
    pub urgent: bool,
    /// Localized tray-tooltip line used for warnings when the overlay is off.
    pub tray_fallback: Option<String>,
}

impl Notice {
    pub fn info(message: NoticeText) -> Self {
        Self {
            kind: NoticeKind::Info,
            message,
            action: None,
            duration: INFO_DURATION,
            urgent: false,
            tray_fallback: None,
        }
    }

    pub fn warning(message: NoticeText) -> Self {
        Self {
            kind: NoticeKind::Warning,
            message,
            action: None,
            duration: WARNING_DURATION,
            urgent: false,
            tray_fallback: None,
        }
    }

    pub fn with_action(mut self, action: NoticeAction) -> Self {
        self.action = Some(action);
        self
    }

    // Part of the shared notice API; not every notice overrides its duration.
    #[allow(dead_code)]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    pub fn urgent(mut self) -> Self {
        self.urgent = true;
        self
    }

    pub fn with_tray_fallback(mut self, text: impl Into<String>) -> Self {
        self.tray_fallback = Some(text.into());
        self
    }
}

/// Payload of the `overlay-notice` event consumed by `RecordingOverlay.tsx`.
#[derive(Clone, Debug, Serialize, Type)]
pub struct OverlayNoticePayload {
    pub id: u64,
    pub kind: NoticeKind,
    pub message: NoticeText,
    pub action: Option<NoticeText>,
    pub duration_ms: u64,
    pub urgent: bool,
}

struct ActiveNotice {
    id: u64,
    action: Option<NoticeHandler>,
}

static NEXT_NOTICE_ID: AtomicU64 = AtomicU64::new(1);
static ACTIVE_NOTICE: Mutex<Option<ActiveNotice>> = Mutex::new(None);
/// True while a warning's tray-tooltip fallback is displayed.
static TRAY_FALLBACK_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The notice the overlay shows, while it shows one (0 = none). Unlike
/// `ACTIVE_NOTICE` it survives the action being taken, so a stale notice
/// can always be dismissed.
static SHOWN_NOTICE: AtomicU64 = AtomicU64::new(0);

fn active_notice() -> std::sync::MutexGuard<'static, Option<ActiveNotice>> {
    ACTIVE_NOTICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Show `notice` in the recording overlay, replacing any notice already shown.
/// Returns the notice id, or `None` when the overlay is off (info dropped,
/// warning sent to the tray tooltip if it has a fallback line).
pub fn show_overlay_notice(app: &AppHandle, notice: Notice) -> Option<u64> {
    if settings::get_settings(app).overlay_style == OverlayStyle::None {
        if notice.kind == NoticeKind::Warning {
            if let Some(line) = notice.tray_fallback {
                TRAY_FALLBACK_ACTIVE.store(true, Ordering::Release);
                crate::tray::set_tray_notice(app, Some(line));
            }
        }
        return None;
    }

    let id = NEXT_NOTICE_ID.fetch_add(1, Ordering::AcqRel);
    let payload = OverlayNoticePayload {
        id,
        kind: notice.kind,
        message: notice.message,
        action: notice.action.as_ref().map(|a| a.label.clone()),
        duration_ms: notice.duration.as_millis() as u64,
        urgent: notice.urgent,
    };
    *active_notice() = Some(ActiveNotice {
        id,
        action: notice.action.map(|a| a.run),
    });
    debug!("overlay notice {id}: {}", payload.message.key);
    SHOWN_NOTICE.store(id, Ordering::Release);
    crate::overlay::show_notice_overlay(app, payload);

    // The overlay webview owns the visible timer (it pauses on hover) and
    // reports back through `dismiss_overlay_notice`; this is only a backstop.
    let handle = app.clone();
    let backstop = notice.duration + MAX_HOVER_EXTENSION;
    std::thread::spawn(move || {
        std::thread::sleep(backstop);
        dismiss_notice(&handle, id);
    });

    Some(id)
}

/// Called when the overlay switches to a dictation state: a new press always
/// wins, so the current notice (and its pending action) is dropped.
pub(crate) fn forget_active_notice() {
    active_notice().take();
    SHOWN_NOTICE.store(0, Ordering::Release);
}

/// Called at the start of every dictation press. Clears a warning's tray
/// tooltip fallback (cheap no-op when none is shown).
pub fn on_new_press(app: &AppHandle) {
    if TRAY_FALLBACK_ACTIVE.swap(false, Ordering::AcqRel) {
        crate::tray::set_tray_notice(app, None);
    }
}

/// Hide the overlay if notice `id` is still the one showing.
fn dismiss_notice(app: &AppHandle, id: u64) {
    let still_showing = {
        let mut active = active_notice();
        if active.as_ref().is_some_and(|n| n.id == id) {
            active.take();
            true
        } else {
            // Its action already ran (or was dropped) but the overlay may
            // still show it: hide only if nothing replaced it since.
            active.is_none() && SHOWN_NOTICE.load(Ordering::Acquire) == id
        }
    };
    if still_showing {
        SHOWN_NOTICE.store(0, Ordering::Release);
        crate::overlay::hide_recording_overlay(app);
    }
}

/// The overlay's notice timer ran out (pauses while hovered).
#[tauri::command]
#[specta::specta]
pub fn dismiss_overlay_notice(app: AppHandle, id: u64) {
    dismiss_notice(&app, id);
}

/// The user clicked the action button of notice `id`.
#[tauri::command]
#[specta::specta]
pub fn run_overlay_notice_action(app: AppHandle, id: u64) {
    let (handler, running) = {
        let mut active = active_notice();
        match active.as_mut() {
            Some(notice) if notice.id == id => {
                let handler = notice.action.take();
                let running = handler.is_none();
                (handler, running)
            }
            _ => (None, false),
        }
    };
    let Some(handler) = handler else {
        debug!("overlay notice {id}: action no longer available");
        if !running {
            // Nothing will ever finish this notice: un-freeze the overlay
            // (its timer resumes) and hide it if it is still showing.
            let _ = app.emit_to("recording_overlay", "overlay-notice-action-done", id);
            dismiss_notice(&app, id);
        }
        return;
    };
    // Handlers may block (device switches, settings I/O); keep them off the
    // IPC/main thread.
    std::thread::spawn(move || {
        handler(&app);
        // No follow-up notice and no new press in the meantime: we're done.
        dismiss_notice(&app, id);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_spec_durations() {
        let info = Notice::info(NoticeText::new("a"));
        assert_eq!(info.kind, NoticeKind::Info);
        assert_eq!(info.duration, Duration::from_millis(1_500));
        assert!(!info.urgent);

        let warning = Notice::warning(NoticeText::new("b"))
            .urgent()
            .with_action(NoticeAction::new(NoticeText::new("c"), |_| {}));
        assert_eq!(warning.kind, NoticeKind::Warning);
        assert_eq!(warning.duration, Duration::from_millis(4_000));
        assert!(warning.urgent);
        assert_eq!(warning.action.map(|a| a.label.key), Some("c".to_string()));
    }

    #[test]
    fn notice_text_serializes_key_and_params() {
        let text = NoticeText::new("overlay.notice.heardNothing").param("mic", "USB Mic");
        let json = serde_json::to_value(&text).expect("serialize");
        assert_eq!(json["key"], "overlay.notice.heardNothing");
        assert_eq!(json["params"]["mic"], "USB Mic");
    }
}
