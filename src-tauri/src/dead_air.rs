//! Dead-air guard: tell the user when a recording heard nothing.
//!
//! After a recording stops, `actions.rs` classifies it with
//! [`crate::audio_toolkit::classify_clip`] (pre-VAD level stats gathered in the
//! capture consumer, so the normal path pays nothing extra). A dead clip skips
//! transcription, cleanup, paste and History and shows an overlay warning naming
//! the microphone, with a one-click switch to the last other mic that produced
//! speech. Three silent clips in a row from the same mic raise a persistent
//! alert (tray warning icon + menu item, one OS notification per session, and
//! an Alert on General > Sound) that clears on the next clip with speech, on a
//! device change, or when dismissed.

use crate::audio_toolkit::{list_input_devices, ClipVerdict};
use crate::managers::audio::AudioRecordingManager;
use crate::overlay_notice::{show_overlay_notice, Notice, NoticeAction, NoticeText};
use crate::settings::{get_settings, write_settings};
use crate::tray_i18n::get_tray_translations;
use log::{debug, error, info, warn};
use serde::Serialize;
use specta::Type;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// Consecutive silent clips that raise the persistent alert. Tolerates the
/// "pressed, then thought better of it" case and a Bluetooth headset's first
/// silent clip while it renegotiates.
pub const SILENT_STREAK_ALERT: u32 = 3;

/// Known-good microphones remembered for the "Use {{mic}}" action.
const MAX_GOOD_MICS: usize = 5;

/// Event carrying `Option<SilentMicAlert>` whenever the persistent alert changes.
pub const SILENT_MIC_ALERT_EVENT: &str = "silent-mic-alert-changed";

/// The persistent "microphone keeps recording silence" alert.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Type)]
pub struct SilentMicAlert {
    pub mic: String,
    pub count: u32,
}

#[derive(Default)]
struct GuardState {
    streak: u32,
    streak_mic: Option<String>,
    /// Most recent first.
    good_mics: VecDeque<String>,
    alert: Option<SilentMicAlert>,
    os_notified: bool,
}

impl GuardState {
    /// A clip that was not dead air. Ends the silent streak; a clip with
    /// speech also clears the alert and remembers its mic as known-good.
    /// Returns true when the alert was cleared.
    fn record_usable(&mut self, mic: Option<&str>, had_speech: bool) -> bool {
        self.streak = 0;
        self.streak_mic = None;
        if !had_speech {
            return false;
        }
        if let Some(mic) = mic {
            self.good_mics.retain(|m| m != mic);
            self.good_mics.push_front(mic.to_string());
            self.good_mics.truncate(MAX_GOOD_MICS);
        }
        self.alert.take().is_some()
    }

    /// A dead clip from `mic`. Returns the alert when this clip raised or
    /// updated it.
    fn record_silent(&mut self, mic: &str) -> Option<SilentMicAlert> {
        if self.streak_mic.as_deref() == Some(mic) {
            self.streak += 1;
        } else {
            self.streak_mic = Some(mic.to_string());
            self.streak = 1;
        }
        if self.streak >= SILENT_STREAK_ALERT {
            let alert = SilentMicAlert {
                mic: mic.to_string(),
                count: self.streak,
            };
            self.alert = Some(alert.clone());
            Some(alert)
        } else {
            None
        }
    }

    /// Most recent known-good mic other than `current`.
    fn fallback_mic(&self, current: &str) -> Option<&str> {
        self.good_mics
            .iter()
            .map(String::as_str)
            .find(|m| *m != current)
    }

    /// Device change, dismiss, or guard turned off. Returns true when an
    /// alert was showing.
    fn reset(&mut self) -> bool {
        self.streak = 0;
        self.streak_mic = None;
        self.alert.take().is_some()
    }
}

static STATE: Mutex<Option<GuardState>> = Mutex::new(None);

fn with_state<R>(f: impl FnOnce(&mut GuardState) -> R) -> R {
    let mut guard: MutexGuard<'_, Option<GuardState>> =
        STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(GuardState::default))
}

fn publish_alert(app: &AppHandle, alert: Option<SilentMicAlert>) {
    crate::tray::set_mic_silent_alert(app, alert.as_ref().map(|a| a.mic.clone()));
    let _ = app.emit(SILENT_MIC_ALERT_EVENT, alert);
}

/// Called for every finished clip that was not dead air (cheap: one lock).
pub fn on_usable_clip(app: &AppHandle, mic: Option<&str>, had_speech: bool) {
    if with_state(|s| s.record_usable(mic, had_speech)) {
        info!("Silent-mic alert cleared: a clip contained speech");
        publish_alert(app, None);
    }
}

/// Called when the selected microphone changes or the guard is turned off.
pub fn reset(app: &AppHandle) {
    if with_state(GuardState::reset) {
        publish_alert(app, None);
    }
}

/// Handle a dead clip: warn in the overlay and track the silent streak.
/// The caller has already skipped transcription and torn down any stream.
///
/// Runs in the stop task while the coordinator is still busy, so it must stay
/// cheap: no device enumeration here (the "Use {{mic}}" action checks that
/// the mic still exists when it is clicked).
pub fn on_silent_clip(app: &AppHandle, verdict: ClipVerdict, mic: Option<String>) {
    let settings = get_settings(app);
    let strings = get_tray_translations(Some(settings.app_language.clone()));
    let mic = mic
        .or(settings.selected_microphone)
        .unwrap_or_else(|| strings.default_microphone.clone());

    let (alert, fallback) = with_state(|s| {
        let alert = s.record_silent(&mic);
        (alert, s.fallback_mic(&mic).map(str::to_string))
    });

    let no_audio = verdict == ClipVerdict::NoAudio;
    let (message_key, tray_template) = if no_audio {
        let key = if cfg!(target_os = "windows") {
            "overlay.notice.sentNoAudioWindows"
        } else {
            "overlay.notice.sentNoAudio"
        };
        (key, strings.sent_no_audio.clone())
    } else {
        ("overlay.notice.heardNothing", strings.heard_nothing.clone())
    };

    let action = match fallback {
        Some(other) => NoticeAction::new(
            NoticeText::new("overlay.notice.useMic").param("mic", other.clone()),
            move |app| switch_microphone(app, &other),
        ),
        None => NoticeAction::new(NoticeText::new("overlay.notice.micSettings"), |app| {
            open_sound_settings(app)
        }),
    };
    show_overlay_notice(
        app,
        Notice::warning(NoticeText::new(message_key).param("mic", mic.clone()))
            .urgent()
            .with_action(action)
            .with_tray_fallback(tray_template.replace("{{mic}}", &mic)),
    );

    if let Some(alert) = alert {
        warn!(
            "Silent-mic alert: last {} recordings from '{}' were silent",
            alert.count, alert.mic
        );
        let notify = with_state(|s| !std::mem::replace(&mut s.os_notified, true));
        if notify {
            send_os_notification(app, &alert);
        }
        publish_alert(app, Some(alert));
    }
}

fn send_os_notification(app: &AppHandle, alert: &SilentMicAlert) {
    use tauri_plugin_notification::NotificationExt;

    let strings = get_tray_translations(Some(get_settings(app).app_language));
    let body = strings
        .mic_silent_notification_body
        .replace("{{mic}}", &alert.mic)
        .replace("{{count}}", &alert.count.to_string());
    if let Err(err) = app
        .notification()
        .builder()
        .title(&strings.mic_silent_notification_title)
        .body(body)
        .show()
    {
        warn!("Failed to send silent-mic notification: {err}");
    }
}

/// Open the main window on General (which hosts the Sound settings).
pub fn open_sound_settings(app: &AppHandle) {
    crate::show_main_window(app);
    let _ = app.emit("navigate-to-section", "general");
}

/// Longest a "Use {{mic}}" click waits for an in-flight dictation to finish.
const SWITCH_WAIT_LIMIT: Duration = Duration::from_secs(60);
const SWITCH_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// True while a dictation is recording or being processed.
fn dictation_active(app: &AppHandle) -> bool {
    let coordinator_busy = app
        .try_state::<crate::TranscriptionCoordinator>()
        .is_some_and(|c| c.is_busy());
    let recording = app
        .try_state::<Arc<AudioRecordingManager>>()
        .is_some_and(|a| a.is_recording());
    coordinator_busy || recording
}

/// Make `name` the selected microphone (the "Use {{mic}}" action).
///
/// Switching restarts the capture stream, which would cut off a recording in
/// progress (and its empty stop would be charged to the new mic), so a click
/// during a dictation is deferred until the pipeline is idle.
fn switch_microphone(app: &AppHandle, name: &str) {
    if dictation_active(app) {
        let app = app.clone();
        let name = name.to_string();
        let spawned = std::thread::Builder::new()
            .name("dead-air-mic-switch".to_string())
            .spawn(move || {
                let started = Instant::now();
                while dictation_active(&app) {
                    if started.elapsed() >= SWITCH_WAIT_LIMIT {
                        warn!("Gave up switching to '{name}': dictation still running");
                        return;
                    }
                    std::thread::sleep(SWITCH_POLL_INTERVAL);
                }
                apply_microphone_switch(&app, &name);
            });
        if let Err(err) = spawned {
            error!("Failed to defer microphone switch: {err}");
        }
        return;
    }
    apply_microphone_switch(app, name);
}

fn apply_microphone_switch(app: &AppHandle, name: &str) {
    // The mic was known-good earlier but may have been unplugged since.
    let present = list_input_devices()
        .map(|devices| devices.iter().any(|d| d.name == name))
        .unwrap_or(false);
    if !present {
        warn!("Microphone '{name}' is no longer available");
        show_overlay_notice(
            app,
            Notice::warning(NoticeText::new("overlay.notice.switchFailed").param("mic", name)),
        );
        return;
    }

    let mut settings = get_settings(app);
    settings.selected_microphone = Some(name.to_string());
    write_settings(app, settings);
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "selected_microphone", "value": name }),
    );
    reset(app);

    let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
    match rm.update_selected_device() {
        Ok(()) => {
            debug!("Switched microphone to '{name}' from the dead-air notice");
            show_overlay_notice(
                app,
                Notice::info(NoticeText::new("overlay.notice.switchedMic").param("mic", name)),
            );
        }
        Err(err) => {
            error!("Failed to switch microphone to '{name}': {err}");
            show_overlay_notice(
                app,
                Notice::warning(NoticeText::new("overlay.notice.switchFailed").param("mic", name)),
            );
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_silent_mic_alert() -> Option<SilentMicAlert> {
    with_state(|s| s.alert.clone())
}

#[tauri::command]
#[specta::specta]
pub fn dismiss_silent_mic_alert(app: AppHandle) {
    reset(&app);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_consecutive_silent_clips_raise_the_alert() {
        let mut state = GuardState::default();
        assert_eq!(state.record_silent("USB"), None);
        assert_eq!(state.record_silent("USB"), None);
        assert_eq!(
            state.record_silent("USB"),
            Some(SilentMicAlert {
                mic: "USB".into(),
                count: 3
            })
        );
        // Further silent clips keep it up with the running count.
        assert_eq!(state.record_silent("USB").map(|a| a.count), Some(4));
    }

    #[test]
    fn a_usable_clip_breaks_the_streak() {
        let mut state = GuardState::default();
        state.record_silent("USB");
        state.record_silent("USB");
        assert!(!state.record_usable(Some("USB"), false));
        assert_eq!(state.record_silent("USB"), None);
    }

    #[test]
    fn switching_mics_restarts_the_streak() {
        let mut state = GuardState::default();
        state.record_silent("USB");
        state.record_silent("USB");
        assert_eq!(state.record_silent("Laptop"), None);
    }

    #[test]
    fn speech_clears_the_alert_but_noise_does_not() {
        let mut state = GuardState::default();
        for _ in 0..3 {
            state.record_silent("USB");
        }
        assert!(state.alert.is_some());
        // A non-dead clip without speech ends the streak but keeps the alert.
        assert!(!state.record_usable(Some("USB"), false));
        assert!(state.alert.is_some());
        assert!(state.record_usable(Some("USB"), true));
        assert!(state.alert.is_none());
    }

    #[test]
    fn reset_clears_alert_and_streak() {
        let mut state = GuardState::default();
        for _ in 0..3 {
            state.record_silent("USB");
        }
        assert!(state.reset());
        assert!(!state.reset());
        assert_eq!(state.record_silent("USB"), None);
    }

    #[test]
    fn fallback_is_the_most_recent_other_good_mic() {
        let mut state = GuardState::default();
        assert_eq!(state.fallback_mic("USB"), None);
        state.record_usable(Some("Laptop"), true);
        state.record_usable(Some("Headset"), true);
        state.record_usable(Some("USB"), true);
        assert_eq!(state.fallback_mic("USB"), Some("Headset"));
        assert_eq!(state.fallback_mic("Headset"), Some("USB"));
        // Only the current mic is known-good: nothing else to offer.
        let mut only = GuardState::default();
        only.record_usable(Some("USB"), true);
        assert_eq!(only.fallback_mic("USB"), None);
    }

    #[test]
    fn good_mic_list_is_bounded_and_deduplicated() {
        let mut state = GuardState::default();
        for i in 0..10 {
            state.record_usable(Some(&format!("mic{i}")), true);
        }
        state.record_usable(Some("mic9"), true);
        assert_eq!(state.good_mics.len(), MAX_GOOD_MICS);
        assert_eq!(state.good_mics.front().map(String::as_str), Some("mic9"));
    }
}
