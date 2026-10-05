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
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::AppHandle;

/// A pressed second tap that never resolves (cancelled, failed to start) is
/// forgotten after this long so it cannot block the first tap forever.
const STALE_SECOND_PRESS: Duration = Duration::from_secs(30);

/// Allowed tap length (ms), matching the settings slider.
pub const TAP_MAX_RANGE_MS: RangeInclusive<u64> = 100..=400;
/// Allowed double-tap window (ms), matching the settings slider.
pub const DOUBLE_TAP_RANGE_MS: RangeInclusive<u64> = 150..=600;

fn clamp_ms(ms: u64, range: &RangeInclusive<u64>) -> u64 {
    ms.clamp(*range.start(), *range.end())
}

pub fn clamp_tap_max_ms(ms: u64) -> u64 {
    clamp_ms(ms, &TAP_MAX_RANGE_MS)
}

pub fn clamp_double_tap_ms(ms: u64) -> u64 {
    clamp_ms(ms, &DOUBLE_TAP_RANGE_MS)
}

/// `at + d`, saturating instead of panicking on overflow.
fn after(at: Instant, d: Duration) -> Instant {
    at.checked_add(d).unwrap_or(at)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GestureTiming {
    pub tap_max: Duration,
    pub double_window: Duration,
}

impl GestureTiming {
    /// Timing from settings, clamped to the supported ranges (a hand-edited
    /// settings file can never overflow the timer arithmetic).
    pub fn from_settings(settings: &AppSettings) -> Self {
        Self::from_ms(settings.tap_max_duration_ms, settings.double_tap_window_ms)
    }

    fn from_ms(tap_max_ms: u64, double_window_ms: u64) -> Self {
        Self {
            tap_max: Duration::from_millis(clamp_tap_max_ms(tap_max_ms)),
            double_window: Duration::from_millis(clamp_double_tap_ms(double_window_ms)),
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
    /// A busy-pipeline tap that arrived before the previous tap was
    /// classified; it may still complete a double-tap with it.
    early_second: Option<Tap>,
}

impl GestureMachine {
    /// Is the press `duration` long, with or without speech, a tap?
    pub fn is_tap(duration: Duration, has_speech: bool, timing: GestureTiming) -> bool {
        !has_speech && duration < timing.tap_max
    }

    /// A press of the main binding started.
    pub fn on_press(&mut self, at: Instant, timing: GestureTiming) -> Option<GestureAction> {
        let first = self.first_tap?;
        if self.second_press.is_some() {
            // The previous second press never resolved (cancelled, failed to
            // start): the whole gesture is void. Never paste long after.
            debug!("Dropping an unresolved tap gesture");
            self.first_tap = None;
            self.second_press = None;
            return None;
        }
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
            self.ignore_until = Some(after(released_at, timing.double_window));
            self.early_second = None;
            return None;
        }
        self.ignore_until = None;

        if let Some(second) = self.early_second.take() {
            if second.pressed_at >= released_at
                && second.pressed_at.saturating_duration_since(released_at) <= timing.double_window
            {
                self.first_tap = None;
                self.second_press = None;
                self.ignore_until = Some(after(second.released_at, timing.double_window));
                return Some(GestureAction::SwapLast {
                    first_press: pressed_at,
                });
            }
        }

        if let Some(first) = self.first_tap {
            if pressed_at.saturating_duration_since(first.released_at) <= timing.double_window {
                self.first_tap = None;
                self.second_press = None;
                self.ignore_until = Some(after(released_at, timing.double_window));
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
    /// tap was still being torn down). It can only complete a double-tap:
    /// with the waiting first tap, or with the previous press once that is
    /// classified as a tap.
    pub fn on_busy_tap(
        &mut self,
        pressed_at: Instant,
        released_at: Instant,
        timing: GestureTiming,
    ) -> Option<GestureAction> {
        if self.first_tap.is_none() {
            self.early_second = Some(Tap {
                pressed_at,
                released_at,
            });
            return None;
        }
        self.on_tap(pressed_at, released_at, timing)
    }

    /// A finished press turned out to be a real dictation. A first tap that
    /// was waiting stands on its own (its window passed without a 2nd tap).
    pub fn on_dictation(&mut self) -> Option<GestureAction> {
        self.second_press = None;
        self.ignore_until = None;
        self.early_second = None;
        self.first_tap.take().map(|_| GestureAction::PasteLast)
    }

    /// A press was cancelled or never became a recording (start failed, the
    /// pipeline dropped it). Any gesture in progress is void: nothing fires.
    pub fn on_abandoned(&mut self) {
        self.first_tap = None;
        self.second_press = None;
        self.early_second = None;
    }

    /// When [`GestureMachine::on_timer`] should run next.
    pub fn next_deadline(&self, timing: GestureTiming) -> Option<Instant> {
        match (self.first_tap, self.second_press) {
            (Some(_), Some(pressed)) => Some(after(pressed, STALE_SECOND_PRESS)),
            (Some(first), None) => Some(after(first.released_at, timing.double_window)),
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
            None if now >= after(first.released_at, timing.double_window) => {
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

/// The held-back recording overlay of the current press: still pending, shown,
/// or closed (stop began; it must not appear any more).
const OVERLAY_PENDING: u8 = 0;
const OVERLAY_SHOWN: u8 = 1;
const OVERLAY_CLOSED: u8 = 2;
static PRESS_OVERLAY: Mutex<u8> = Mutex::new(OVERLAY_SHOWN);

fn press_overlay() -> std::sync::MutexGuard<'static, u8> {
    PRESS_OVERLAY.lock().unwrap_or_else(|e| e.into_inner())
}

/// A press started; `held_back` when its overlay waits for the tap window
/// (otherwise it was shown right away).
pub fn reset_press_overlay(held_back: bool) {
    *press_overlay() = if held_back {
        OVERLAY_PENDING
    } else {
        OVERLAY_SHOWN
    };
}

/// Show the held-back overlay via `show` (which returns `None` when it
/// decided not to), unless stop already began. Serialised with
/// [`close_press_overlay`], so the recording overlay can never land on top of
/// the working state.
pub fn try_show_press_overlay(show: impl FnOnce() -> Option<()>) -> bool {
    let mut state = press_overlay();
    if *state != OVERLAY_PENDING {
        return false;
    }
    if show().is_some() {
        *state = OVERLAY_SHOWN;
        return true;
    }
    false
}

/// Stop began: the held-back overlay may no longer appear. Returns whether
/// the recording overlay is on screen for this press.
pub fn close_press_overlay() -> bool {
    let mut state = press_overlay();
    let shown = *state == OVERLAY_SHOWN;
    *state = OVERLAY_CLOSED;
    shown
}

/// Gestures apply to this binding with the current settings.
pub fn active_for(binding_id: &str, settings: &AppSettings) -> bool {
    binding_id == "transcribe" && settings.tap_gestures_active()
}

/* Settings cache: the press path runs on every key press, so it must not
 * deserialize the full settings. Refreshed on every settings read/write
 * (see `cockpit::sync_settings`). */

const CACHE_UNKNOWN: u8 = 0;
const CACHE_OFF: u8 = 1;
const CACHE_ON: u8 = 2;
static CACHED_ACTIVE: AtomicU8 = AtomicU8::new(CACHE_UNKNOWN);
static CACHED_TAP_MAX_MS: AtomicU64 = AtomicU64::new(0);
static CACHED_DOUBLE_MS: AtomicU64 = AtomicU64::new(0);

pub fn cache_settings(settings: &AppSettings) {
    CACHED_TAP_MAX_MS.store(settings.tap_max_duration_ms, Ordering::Relaxed);
    CACHED_DOUBLE_MS.store(settings.double_tap_window_ms, Ordering::Relaxed);
    let active = if settings.tap_gestures_active() {
        CACHE_ON
    } else {
        CACHE_OFF
    };
    CACHED_ACTIVE.store(active, Ordering::Release);
}

/// Gesture timing when gestures are active (from the cache).
fn cached_timing(app: &AppHandle) -> Option<GestureTiming> {
    if CACHED_ACTIVE.load(Ordering::Acquire) == CACHE_UNKNOWN {
        cache_settings(&get_settings(app));
    }
    (CACHED_ACTIVE.load(Ordering::Acquire) == CACHE_ON).then(|| {
        GestureTiming::from_ms(
            CACHED_TAP_MAX_MS.load(Ordering::Relaxed),
            CACHED_DOUBLE_MS.load(Ordering::Relaxed),
        )
    })
}

/// Coordinator: a confirmed press of a transcribe binding (auto-repeat and
/// debounced presses filtered out).
pub fn on_main_press(app: &AppHandle, binding_id: &str, at: Instant) {
    if binding_id != "transcribe" {
        return;
    }
    let Some(timing) = cached_timing(app) else {
        return;
    };
    if let Some(action) = with_machine(|m| m.on_press(at, timing)) {
        execute(app, action);
    }
}

/// A press was cancelled or never turned into a classified recording.
pub fn abandon() {
    let mut guard = MACHINE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(machine) = guard.as_mut() {
        machine.on_abandoned();
    }
}

/// Coordinator: a confirmed release of a transcribe binding (after the
/// auto-repeat grace).
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
    if binding_id != "transcribe" {
        return;
    }
    let Some(timing) = cached_timing(app) else {
        return;
    };
    if released_at.saturating_duration_since(pressed_at) >= timing.tap_max {
        // A long hold that never recorded: whatever gesture was in progress
        // is void.
        abandon();
        return;
    }
    if let Some(action) = with_machine(|m| m.on_busy_tap(pressed_at, released_at, timing)) {
        execute(app, action);
    }
}

/// Pipeline: the finished press was a tap.
pub fn on_tap(app: &AppHandle, pressed_at: Instant, released_at: Instant) {
    let timing =
        cached_timing(app).unwrap_or_else(|| GestureTiming::from_settings(&get_settings(app)));
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
        let timing = cached_timing(&app).unwrap_or(timing);
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
    fn busy_tap_before_first_tap_is_classified_still_swaps() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        // Tap 2 is released while tap 1 is still in the pipeline.
        assert_eq!(m.on_busy_tap(ms(t, 200), ms(t, 260), TIMING), None);
        assert_eq!(
            m.on_tap(t, ms(t, 100), TIMING),
            Some(GestureAction::SwapLast { first_press: t })
        );
        assert_eq!(m.next_deadline(TIMING), None);
    }

    #[test]
    fn abandoned_press_voids_the_gesture() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        m.on_press(ms(t, 200), TIMING);
        m.on_abandoned();
        assert_eq!(m.next_deadline(TIMING), None);
        assert_eq!(m.on_timer(ms(t, 5_000), TIMING), None);
    }

    #[test]
    fn unresolved_second_press_never_pastes_on_a_later_press() {
        let t = Instant::now();
        let mut m = GestureMachine::default();
        m.on_tap(t, ms(t, 100), TIMING);
        assert_eq!(m.on_press(ms(t, 200), TIMING), None);
        // That press never resolved; much later the user presses again.
        assert_eq!(m.on_press(ms(t, 20_000), TIMING), None);
        assert_eq!(m.next_deadline(TIMING), None);
    }

    #[test]
    fn timing_is_clamped_to_supported_ranges() {
        let wild = GestureTiming::from_ms(u64::MAX, 0);
        assert_eq!(wild.tap_max, Duration::from_millis(400));
        assert_eq!(wild.double_window, Duration::from_millis(150));
        let t = Instant::now();
        let mut m = GestureMachine::default();
        // No overflow panic even with extreme instants.
        m.on_tap(t, t, wild);
        assert!(m.next_deadline(wild).is_some());
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
