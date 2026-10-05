//! Tap gestures on the main dictation binding (push-to-talk only):
//! a short press with no speech is a *tap* → Paste last; a second tap that
//! starts within the double-tap window of the first release → Swap last.
//!
//! Recording still starts on every press (no audio is lost). The stop
//! pipeline classifies a finished press: short + no speech = tap (discarded:
//! no STT, no History, no dead-air count), anything else is a dictation.
//! [`GestureMachine`] is the pure timing logic; the glue below feeds it from
//! the coordinator (press/release instants) and the pipeline (classification).

use crate::audio_toolkit::audio::{CaptureStats, NOISE_FLOOR_PEAK, NOISE_FLOOR_RMS};
use crate::settings::{get_settings, AppSettings};
use log::debug;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::AppHandle;

/// A pressed second tap that never resolves (cancelled, failed to start) is
/// forgotten after this long so it cannot block the first tap forever.
const STALE_SECOND_PRESS: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GestureTiming {
    pub tap_max: Duration,
    pub double_window: Duration,
}

impl GestureTiming {
    pub fn from_settings(settings: &AppSettings) -> Self {
        Self {
            tap_max: Duration::from_millis(settings.tap_max_duration_ms),
            double_window: Duration::from_millis(settings.double_tap_window_ms),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureAction {
    PasteLast,
    /// `first_press` is when the gesture began: any real input before it
    /// means the user touched something after the paste.
    SwapLast {
        first_press: Instant,
    },
}

#[derive(Clone, Copy, Debug)]
struct Tap {
    pressed_at: Instant,
    released_at: Instant,
}

#[derive(Debug, Default)]
pub struct GestureMachine {
    /// A tap waiting for the double-tap window to expire.
    first_tap: Option<Tap>,
    /// A press that began inside the window and is not classified yet. While
    /// it is held the single-tap action must not fire.
    second_press: Option<Instant>,
    /// Taps that begin before this instant are ignored (3rd+ taps).
    ignore_until: Option<Instant>,
}

impl GestureMachine {
    /// Is the press `duration` long, with or without speech, a tap?
    pub fn is_tap(duration: Duration, has_speech: bool, timing: GestureTiming) -> bool {
        !has_speech && duration < timing.tap_max
    }

    /// A press of the main binding started.
    pub fn on_press(&mut self, at: Instant, timing: GestureTiming) -> Option<GestureAction> {
        let first = self.first_tap?;
        if at.saturating_duration_since(first.released_at) <= timing.double_window {
            self.second_press = Some(at);
            None
        } else {
            // The window already expired but its timer has not run yet: the
            // first tap stands on its own.
            self.first_tap = None;
            self.second_press = None;
            Some(GestureAction::PasteLast)
        }
    }

    /// A finished press was classified as a tap.
    pub fn on_tap(
        &mut self,
        pressed_at: Instant,
        released_at: Instant,
        timing: GestureTiming,
    ) -> Option<GestureAction> {
        if self.ignore_until.is_some_and(|until| pressed_at <= until) {
            debug!("Ignoring extra tap after a double-tap");
            self.ignore_until = Some(released_at + timing.double_window);
            return None;
        }
        self.ignore_until = None;

        if let Some(first) = self.first_tap {
            if pressed_at.saturating_duration_since(first.released_at) <= timing.double_window {
                self.first_tap = None;
                self.second_press = None;
                self.ignore_until = Some(released_at + timing.double_window);
                return Some(GestureAction::SwapLast {
                    first_press: first.pressed_at,
                });
            }
        }

        self.first_tap = Some(Tap {
            pressed_at,
            released_at,
        });
        self.second_press = None;
        None
    }

    /// A tap that never reached the pipeline (released while the previous
    /// tap was still being torn down). It can only complete a double-tap.
    pub fn on_busy_tap(
        &mut self,
        pressed_at: Instant,
        released_at: Instant,
        timing: GestureTiming,
    ) -> Option<GestureAction> {
        self.first_tap?;
        self.on_tap(pressed_at, released_at, timing)
    }

    /// A finished press turned out to be a real dictation. A first tap that
    /// was waiting stands on its own (its window passed without a 2nd tap).
    pub fn on_dictation(&mut self) -> Option<GestureAction> {
        self.second_press = None;
        self.ignore_until = None;
        self.first_tap.take().map(|_| GestureAction::PasteLast)
    }

    /// When [`GestureMachine::on_timer`] should run next.
    pub fn next_deadline(&self, timing: GestureTiming) -> Option<Instant> {
        match (self.first_tap, self.second_press) {
            (Some(_), Some(pressed)) => Some(pressed + STALE_SECOND_PRESS),
            (Some(first), None) => Some(first.released_at + timing.double_window),
            _ => None,
        }
    }

    pub fn on_timer(&mut self, now: Instant, timing: GestureTiming) -> Option<GestureAction> {
        let first = self.first_tap?;
        match self.second_press {
            Some(pressed) => {
                if now.saturating_duration_since(pressed) >= STALE_SECOND_PRESS {
                    self.first_tap = None;
                    self.second_press = None;
                }
                None
            }
            None if now >= first.released_at + timing.double_window => {
                self.first_tap = None;
                Some(GestureAction::PasteLast)
            }
            None => None,
        }
    }
}

/// Speech evidence for tap classification. With VAD off there is no speech
/// signal, so fall back to level: anything above the noise floor counts as
/// speech (speech wins).
pub fn clip_has_speech(stats: &CaptureStats) -> bool {
    if stats.vad_active {
        return stats.has_speech();
    }
    !(stats.peak < NOISE_FLOOR_PEAK && stats.rms() < NOISE_FLOOR_RMS)
}

/* ───────────────────────────── runtime glue ───────────────────────────── */

static MACHINE: Mutex<Option<GestureMachine>> = Mutex::new(None);
/// (pressed_at, released_at) of the main-binding hold that ended the
/// current recording, recorded by the coordinator before it runs Stop.
static LAST_HOLD: Mutex<Option<(Instant, Instant)>> = Mutex::new(None);
/// Latest release of the main binding's key (raw edge, before the grace).
static LAST_RELEASE: Mutex<Option<Instant>> = Mutex::new(None);

fn with_machine<T>(f: impl FnOnce(&mut GestureMachine) -> T) -> T {
    let mut guard = MACHINE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(GestureMachine::default))
}

/// Whether the delayed recording overlay was shown for the current press
/// (a tap must not hide a notice it never replaced).
static PRESS_OVERLAY_SHOWN: AtomicBool = AtomicBool::new(false);

pub fn reset_press_overlay() {
    PRESS_OVERLAY_SHOWN.store(false, Ordering::Release);
}

pub fn mark_press_overlay_shown() {
    PRESS_OVERLAY_SHOWN.store(true, Ordering::Release);
}

pub fn press_overlay_shown() -> bool {
    PRESS_OVERLAY_SHOWN.load(Ordering::Acquire)
}

/// Gestures apply to this binding with the current settings.
pub fn active_for(binding_id: &str, settings: &AppSettings) -> bool {
    binding_id == "transcribe" && settings.tap_gestures_active()
}

/// Coordinator: a raw press edge of a transcribe binding.
pub fn on_main_press(app: &AppHandle, binding_id: &str, at: Instant) {
    let settings = get_settings(app);
    if !active_for(binding_id, &settings) {
        return;
    }
    let timing = GestureTiming::from_settings(&settings);
    if let Some(action) = with_machine(|m| m.on_press(at, timing)) {
        execute(app, action);
    }
}

/// Coordinator: a raw release edge of a transcribe binding.
pub fn on_main_release(binding_id: &str, at: Instant) {
    if binding_id == "transcribe" {
        *LAST_RELEASE.lock().unwrap_or_else(|e| e.into_inner()) = Some(at);
    }
}

/// True when the main key has been released at or after `pressed_at`.
pub fn released_since(pressed_at: Instant) -> bool {
    LAST_RELEASE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some_and(|released| released >= pressed_at)
}

/// Coordinator: the hold that is about to be stopped (None for toggle-style
/// stops, which are never taps).
pub fn note_hold(hold: Option<(Instant, Instant)>) {
    *LAST_HOLD.lock().unwrap_or_else(|e| e.into_inner()) = hold;
}

/// The hold noted for the current stop, if any.
pub fn current_hold() -> Option<(Instant, Instant)> {
    *LAST_HOLD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Coordinator: a short press released while the pipeline was still busy
/// (so it never recorded). Counts only as the second tap of a double-tap.
pub fn on_busy_release(
    app: &AppHandle,
    binding_id: &str,
    pressed_at: Instant,
    released_at: Instant,
) {
    let settings = get_settings(app);
    if !active_for(binding_id, &settings) {
        return;
    }
    let timing = GestureTiming::from_settings(&settings);
    if released_at.saturating_duration_since(pressed_at) >= timing.tap_max {
        return;
    }
    if let Some(action) = with_machine(|m| m.on_busy_tap(pressed_at, released_at, timing)) {
        execute(app, action);
    }
}

/// Pipeline: the finished press was a tap.
pub fn on_tap(app: &AppHandle, pressed_at: Instant, released_at: Instant) {
    let timing = GestureTiming::from_settings(&get_settings(app));
    let action = with_machine(|m| m.on_tap(pressed_at, released_at, timing));
    match action {
        Some(action) => execute(app, action),
        None => schedule_timer(app, timing),
    }
}

/// Pipeline: the finished press was a dictation.
pub fn on_dictation(app: &AppHandle) {
    if let Some(action) = with_machine(|m| m.on_dictation()) {
        execute(app, action);
    }
}

fn schedule_timer(app: &AppHandle, timing: GestureTiming) {
    let Some(deadline) = with_machine(|m| m.next_deadline(timing)) else {
        return;
    };
    let app = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        let timing = GestureTiming::from_settings(&get_settings(&app));
        let action = with_machine(|m| m.on_timer(Instant::now(), timing));
        match action {
            Some(action) => execute(&app, action),
            // Still waiting (e.g. a second press is held): check again later.
            None => schedule_timer(&app, timing),
        }
    });
}

/// Paste/swap block (main-thread paste, modifier wait), so they run on their
/// own thread rather than on the coordinator or an async worker.
fn execute(app: &AppHandle, action: GestureAction) {
    debug!("Tap gesture: {action:?}");
    let app = app.clone();
    std::thread::spawn(move || match action {
        GestureAction::PasteLast => super::paste_last(&app),
        GestureAction::SwapLast { first_press } => super::swap_last(&app, first_press),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMING: GestureTiming = GestureTiming {
        tap_max: Duration::from_millis(200),
        double_window: Duration::from_millis(250),
    };

    fn ms(base: Instant, offset: u64) -> Instant {
        base + Duration::from_millis(offset)
    }

    #[test]
    fn tap_requires_short_press_without_speech() {
        assert!(GestureMachine::is_tap(
            Duration::from_millis(120),
            false,
            TIMING
        ));
        // Speech wins: a 180 ms "yes" is a dictation.
        assert!(!GestureMachine::is_tap(
            Duration::from_millis(180),
            true,
            TIMING
        ));
        // A normal hold is never a tap.
        assert!(!GestureMachine::is_tap(
            Duration::from_millis(200),
            false,
            TIMING
        ));
        assert!(!GestureMachine::is_tap(
            Duration::from_millis(900),
            false,
            TIMING
        ));
    }

    #[test]
    fn single_tap_pastes_when_window_expires() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        assert_eq!(m.on_press(t, TIMING), None);
        assert_eq!(m.on_tap(t, ms(t, 120), TIMING), None);
        assert_eq!(m.next_deadline(TIMING), Some(ms(t, 370)));
        assert_eq!(m.on_timer(ms(t, 300), TIMING), None);
        assert_eq!(
            m.on_timer(ms(t, 370), TIMING),
            Some(GestureAction::PasteLast)
        );
        // Fires once.
        assert_eq!(m.on_timer(ms(t, 500), TIMING), None);
    }

    #[test]
    fn double_tap_swaps_and_suppresses_single_paste() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_press(t, TIMING);
        assert_eq!(m.on_tap(t, ms(t, 100), TIMING), None);
        // Second press 150 ms after the first release, inside the window.
        assert_eq!(m.on_press(ms(t, 250), TIMING), None);
        // The window expires while the second press is still held: no paste.
        assert_eq!(m.on_timer(ms(t, 360), TIMING), None);
        assert_eq!(
            m.on_tap(ms(t, 250), ms(t, 330), TIMING),
            Some(GestureAction::SwapLast { first_press: t })
        );
        assert_eq!(m.next_deadline(TIMING), None);
    }

    #[test]
    fn second_press_outside_window_is_a_new_first_tap() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        // Timer has not run yet; the late press flushes the single tap.
        assert_eq!(
            m.on_press(ms(t, 400), TIMING),
            Some(GestureAction::PasteLast)
        );
        assert_eq!(m.on_tap(ms(t, 400), ms(t, 480), TIMING), None);
        assert_eq!(
            m.on_timer(ms(t, 730), TIMING),
            Some(GestureAction::PasteLast)
        );
    }

    #[test]
    fn third_tap_inside_window_is_ignored() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        m.on_press(ms(t, 200), TIMING);
        assert!(matches!(
            m.on_tap(ms(t, 200), ms(t, 280), TIMING),
            Some(GestureAction::SwapLast { .. })
        ));
        // Third tap starts 100 ms after the double-tap ended.
        assert_eq!(m.on_press(ms(t, 380), TIMING), None);
        assert_eq!(m.on_tap(ms(t, 380), ms(t, 450), TIMING), None);
        assert_eq!(m.on_timer(ms(t, 2_000), TIMING), None);
        // A tap well after the window counts again.
        assert_eq!(m.on_tap(ms(t, 2_000), ms(t, 2_080), TIMING), None);
        assert_eq!(
            m.on_timer(ms(t, 2_330), TIMING),
            Some(GestureAction::PasteLast)
        );
    }

    #[test]
    fn dictation_after_tap_releases_the_pending_paste() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        m.on_press(ms(t, 200), TIMING);
        assert_eq!(m.on_dictation(), Some(GestureAction::PasteLast));
        assert_eq!(m.next_deadline(TIMING), None);
        // A plain dictation with nothing pending does nothing.
        assert_eq!(m.on_dictation(), None);
    }

    #[test]
    fn busy_tap_only_completes_a_double_tap() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        assert_eq!(m.on_busy_tap(t, ms(t, 80), TIMING), None);
        assert_eq!(m.next_deadline(TIMING), None, "a lone busy tap is ignored");
        m.on_tap(ms(t, 1_000), ms(t, 1_100), TIMING);
        assert!(matches!(
            m.on_busy_tap(ms(t, 1_200), ms(t, 1_260), TIMING),
            Some(GestureAction::SwapLast { .. })
        ));
    }

    #[test]
    fn stale_second_press_is_forgotten() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        m.on_press(ms(t, 200), TIMING);
        assert_eq!(m.on_timer(ms(t, 200) + STALE_SECOND_PRESS, TIMING), None);
        assert_eq!(m.next_deadline(TIMING), None);
    }

    #[test]
    fn level_decides_speech_when_vad_is_off() {
        let quiet = CaptureStats {
            total_samples: 1_600,
            peak: 0.005,
            sum_squares: 0.0,
            ..Default::default()
        };
        assert!(!clip_has_speech(&quiet));
        let loud = CaptureStats {
            total_samples: 1_600,
            peak: 0.3,
            sum_squares: 1_600.0 * 0.01,
            ..Default::default()
        };
        assert!(clip_has_speech(&loud));
        let vad_silent = CaptureStats {
            vad_active: true,
            ..loud
        };
        assert!(!clip_has_speech(&vad_silent));
    }
}
