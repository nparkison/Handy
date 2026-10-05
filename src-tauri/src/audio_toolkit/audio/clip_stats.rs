//! Cheap per-recording signal statistics and the dead-air classification
//! built on them.
//!
//! The capture consumer folds every 16 kHz frame into a [`CaptureStats`]
//! *before* VAD filtering (one pass of max/abs + sum of squares per frame), so
//! the stats describe what the microphone actually delivered even when VAD
//! later drops every frame. [`classify_clip`] is a pure function over those
//! numbers so the thresholds are unit-testable without audio hardware.

use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

/// Clips shorter than this are misfires (accidental taps), never dead air.
pub const MIN_CLASSIFIED_CLIP_MS: u64 = 300;

/// A press this long that delivered no samples at all means the device is
/// not producing audio. Shorter presses may just have ended before a slow
/// (Bluetooth/USB) device delivered its first callback.
pub const NO_AUDIO_MIN_WALL_MS: u64 = 1_500;

/// Peak at or below ~2 LSB of 16-bit audio (about -84 dBFS) is digital
/// silence: a muted/blocked device or OS privacy block delivering zeros.
pub const DIGITAL_SILENCE_PEAK: f32 = 2.0 / 32_768.0;

/// Peak below about -36 dBFS. Normal speech peaks well above -20 dBFS.
pub const NOISE_FLOOR_PEAK: f32 = 0.016;

/// RMS below about -50 dBFS: a quiet room with nobody talking.
pub const NOISE_FLOOR_RMS: f32 = 0.003_2;

/// Signal statistics for one recording, accumulated over every captured
/// 16 kHz frame before VAD filtering.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CaptureStats {
    /// 16 kHz samples captured (pre-VAD), including pre-press audio.
    pub total_samples: u64,
    /// Portion of `total_samples` that came from the press-time pre-roll.
    /// It counts toward level and speech stats (real audio, possibly the
    /// first word) but not toward the press duration used for tap and
    /// no-audio detection.
    pub pre_roll_samples: u64,
    /// Samples VAD forwarded as speech. Only meaningful when `vad_active`.
    pub speech_samples: u64,
    /// True when a VAD actually evaluated the frames of this recording.
    pub vad_active: bool,
    /// Largest absolute sample value.
    pub peak: f32,
    /// Sum of squared sample values (for RMS).
    pub sum_squares: f64,
}

impl CaptureStats {
    /// Fold one captured frame into the running totals.
    #[inline]
    pub fn observe_frame(&mut self, frame: &[f32]) {
        let mut peak = self.peak;
        let mut sum_squares = 0.0f64;
        for &sample in frame {
            let magnitude = sample.abs();
            if magnitude > peak {
                peak = magnitude;
            }
            sum_squares += f64::from(sample) * f64::from(sample);
        }
        self.peak = peak;
        self.sum_squares += sum_squares;
        self.total_samples += frame.len() as u64;
    }

    /// Samples captured after the press (excludes pre-roll).
    pub fn live_samples(&self) -> u64 {
        self.total_samples.saturating_sub(self.pre_roll_samples)
    }

    /// Duration of audio captured after the press, in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.live_samples() * 1_000 / WHISPER_SAMPLE_RATE as u64
    }

    /// Root-mean-square level of the captured audio (0 when nothing was captured).
    pub fn rms(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        (self.sum_squares / self.total_samples as f64).sqrt() as f32
    }

    /// True when VAD ran and detected speech in this recording.
    pub fn has_speech(&self) -> bool {
        self.vad_active && self.speech_samples > 0
    }
}

/// What a finished recording looks like from the dead-air guard's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipVerdict {
    /// Contains signal worth transcribing (or at least not provably dead).
    Usable,
    /// Too short to judge: a tap or misfire. Handled like any other clip.
    TooShort,
    /// The device delivered digital zeros, or nothing at all. Points at OS
    /// microphone access / a hardware mute rather than at the user.
    NoAudio,
    /// Real (non-zero) input, VAD found no speech, and the level is at the
    /// noise floor. Only produced when VAD ran: level alone can't tell a quiet
    /// room from a soft speaker or a low-gain mic.
    NoSpeech,
}

impl ClipVerdict {
    /// Dead air: skip transcription and tell the user which mic to check.
    pub fn is_dead_air(self) -> bool {
        matches!(self, ClipVerdict::NoAudio | ClipVerdict::NoSpeech)
    }
}

/// Whether a clip the dead-air guard skips still holds real audio (it reached
/// the recording buffer and is above digital silence). Such a clip is saved to
/// History with an empty transcription so it can be retried, instead of being
/// discarded. With VAD on, a skipped clip's buffer is empty (VAD dropped every
/// frame), so this only matters for edge cases such as pre-roll audio
/// followed by a device stall.
pub fn worth_keeping(stats: &CaptureStats, recorded_samples: usize) -> bool {
    recorded_samples > 0 && stats.peak > DIGITAL_SILENCE_PEAK
}

/// Classify a finished recording.
///
/// `wall_ms` is how long recording was requested for (press to release); it
/// only matters when the device delivered no samples at all.
///
/// Rules (thresholds above):
/// Durations count only audio captured after the press; pre-roll audio feeds
/// the peak/RMS/speech checks.
///
/// - No samples at all after a press of at least [`NO_AUDIO_MIN_WALL_MS`]: `NoAudio`.
/// - Under [`MIN_CLASSIFIED_CLIP_MS`] of captured audio: `TooShort`.
/// - Peak at or below [`DIGITAL_SILENCE_PEAK`]: `NoAudio` (digital zeros).
/// - VAD ran, found no speech, and peak or RMS is at the noise floor: `NoSpeech`.
///   (VAD already dropped every frame, so nothing transcribable is lost.)
/// - VAD did not run: never `NoSpeech`. Without a speech signal, a quiet clip
///   may be soft speech, and skipping it would lose the dictation for good.
/// - Otherwise `Usable`.
pub fn classify_clip(stats: &CaptureStats, wall_ms: u64) -> ClipVerdict {
    if stats.live_samples() == 0 {
        return if wall_ms >= NO_AUDIO_MIN_WALL_MS {
            ClipVerdict::NoAudio
        } else {
            ClipVerdict::TooShort
        };
    }
    if stats.duration_ms() < MIN_CLASSIFIED_CLIP_MS {
        return ClipVerdict::TooShort;
    }
    if stats.peak <= DIGITAL_SILENCE_PEAK {
        return ClipVerdict::NoAudio;
    }

    let quiet = stats.peak < NOISE_FLOOR_PEAK || stats.rms() < NOISE_FLOOR_RMS;
    if stats.vad_active && stats.speech_samples == 0 && quiet {
        ClipVerdict::NoSpeech
    } else {
        ClipVerdict::Usable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = WHISPER_SAMPLE_RATE as usize;

    fn stats_for(samples: &[f32], vad_active: bool, speech_samples: u64) -> CaptureStats {
        let mut stats = CaptureStats {
            vad_active,
            speech_samples,
            ..CaptureStats::default()
        };
        for frame in samples.chunks(480) {
            stats.observe_frame(frame);
        }
        stats
    }

    fn tone(amplitude: f32, seconds: f32) -> Vec<f32> {
        let len = (RATE as f32 * seconds) as usize;
        (0..len)
            .map(|i| {
                amplitude * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / RATE as f32).sin()
            })
            .collect()
    }

    #[test]
    fn observe_frame_tracks_peak_rms_and_duration() {
        let stats = stats_for(&[0.5, -1.0, 0.0, 0.5], false, 0);
        assert_eq!(stats.total_samples, 4);
        assert_eq!(stats.peak, 1.0);
        assert!((stats.rms() - (1.5f32 / 4.0).sqrt()).abs() < 1e-6);

        let one_second = stats_for(&vec![0.0; RATE], false, 0);
        assert_eq!(one_second.duration_ms(), 1_000);
    }

    #[test]
    fn digital_zeros_are_no_audio() {
        let stats = stats_for(&vec![0.0; RATE * 2], true, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoAudio);
        // Also without VAD.
        let stats = stats_for(&vec![0.0; RATE * 2], false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoAudio);
    }

    #[test]
    fn one_lsb_dither_still_counts_as_digital_silence() {
        let lsb = 1.0 / 32_768.0;
        let samples: Vec<f32> = (0..RATE)
            .map(|i| if i % 2 == 0 { lsb } else { -lsb })
            .collect();
        let stats = stats_for(&samples, true, 0);
        assert_eq!(classify_clip(&stats, 1_000), ClipVerdict::NoAudio);
    }

    #[test]
    fn no_samples_after_a_long_press_is_no_audio() {
        let stats = CaptureStats {
            vad_active: true,
            ..CaptureStats::default()
        };
        assert_eq!(classify_clip(&stats, 3_000), ClipVerdict::NoAudio);
        // A short press may simply have ended before a slow device started.
        assert_eq!(classify_clip(&stats, 800), ClipVerdict::TooShort);
    }

    #[test]
    fn taps_under_300ms_are_never_dead_air() {
        let stats = stats_for(&vec![0.0; RATE / 4], true, 0); // 250 ms of zeros
        assert_eq!(classify_clip(&stats, 250), ClipVerdict::TooShort);
        assert!(!classify_clip(&stats, 250).is_dead_air());
    }

    #[test]
    fn quiet_room_without_speech_is_no_speech() {
        // -60 dBFS hiss: non-zero, but far below speech.
        let stats = stats_for(&tone(0.001, 2.0), true, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoSpeech);
    }

    #[test]
    fn a_click_in_a_quiet_room_without_speech_is_still_no_speech() {
        // A mouse click spikes the peak, but RMS stays at the noise floor.
        let mut samples = tone(0.001, 2.0);
        samples[100] = 0.4;
        let stats = stats_for(&samples, true, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoSpeech);
    }

    #[test]
    fn detected_speech_is_usable_even_when_quiet() {
        let stats = stats_for(&tone(0.01, 2.0), true, 8_000);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);
    }

    #[test]
    fn loud_noise_without_speech_is_not_flagged() {
        // A fan or music: no VAD speech, but well above the noise floor.
        let stats = stats_for(&tone(0.2, 2.0), true, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);
    }

    #[test]
    fn without_vad_only_digital_silence_is_flagged() {
        // -60 dBFS: a quiet room, or a soft speaker on a low-gain mic. With
        // no VAD to tell them apart, it is transcribed rather than dropped.
        let stats = stats_for(&tone(0.001, 2.0), false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);

        // Soft speech: peak about -40 dBFS, RMS about -43 dBFS.
        let stats = stats_for(&tone(0.01, 2.0), false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);

        let mut samples = tone(0.001, 2.0);
        samples[100] = 0.4;
        let stats = stats_for(&samples, false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);

        let stats = stats_for(&tone(0.3, 2.0), false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::Usable);

        // Digital zeros and a dead device are still caught.
        let stats = stats_for(&vec![0.0; RATE * 2], false, 0);
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoAudio);
        let stats = CaptureStats::default();
        assert_eq!(classify_clip(&stats, 3_000), ClipVerdict::NoAudio);
    }

    #[test]
    fn skipped_clips_keep_real_audio() {
        // Digital zeros hold nothing worth keeping.
        let zeros = stats_for(&vec![0.0; RATE], false, 0);
        assert!(!worth_keeping(&zeros, RATE));
        // Nothing reached the recording buffer (VAD dropped every frame).
        let quiet = stats_for(&tone(0.001, 1.0), true, 0);
        assert!(!worth_keeping(&quiet, 0));
        // Real signal in the buffer: keep it so it can be retried.
        let pre_roll_only = stats_for(&tone(0.2, 0.5), false, 0);
        assert!(worth_keeping(&pre_roll_only, RATE / 2));
    }

    #[test]
    fn has_speech_requires_an_active_vad() {
        let mut stats = stats_for(&tone(0.3, 1.0), false, 0);
        stats.speech_samples = 100;
        assert!(!stats.has_speech());
        stats.vad_active = true;
        assert!(stats.has_speech());
    }

    #[test]
    fn pre_roll_does_not_lengthen_a_tap() {
        // 500 ms of pre-roll + 200 ms live tap: still a tap.
        let mut stats = stats_for(&tone(0.2, 0.7), true, 0);
        stats.pre_roll_samples = (RATE / 2) as u64;
        assert_eq!(stats.duration_ms(), 200);
        assert_eq!(classify_clip(&stats, 200), ClipVerdict::TooShort);
    }

    #[test]
    fn pre_roll_alone_after_a_long_press_is_no_audio() {
        // The device stalled at the press: only buffered pre-roll arrived.
        let mut stats = stats_for(&tone(0.2, 0.5), true, 0);
        stats.pre_roll_samples = stats.total_samples;
        assert_eq!(classify_clip(&stats, 2_000), ClipVerdict::NoAudio);
    }

    #[test]
    fn speech_in_pre_roll_keeps_the_clip_usable() {
        let mut stats = stats_for(&tone(0.01, 1.5), true, 4_000);
        stats.pre_roll_samples = (RATE / 2) as u64;
        assert_eq!(classify_clip(&stats, 1_000), ClipVerdict::Usable);
    }
}
