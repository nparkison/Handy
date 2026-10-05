//! Press-time pre-roll: a bounded, memory-only rolling window of the most
//! recent audio captured while the microphone stream is open but idle.
//!
//! When a recording starts, the window is handed to the capture path ahead of
//! the live samples so the user's first syllable is not clipped by the gap
//! between pressing the shortcut and the recorder processing `Cmd::Start`.
//! The window only ever holds audio the already-open stream delivered; it never
//! opens the microphone itself and is dropped with the stream.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Upper bound for the user setting. Longer windows start pulling in the tail
/// of a previous sentence or conversation.
pub const MAX_PRE_ROLL_MS: u64 = 1_000;

/// Audio captured this soon after a recording stops never seeds the next
/// pre-roll. It is the stop chime and the tail of a word the user was still
/// saying at release, which a quick re-press would otherwise prepend.
pub const POST_STOP_GUARD_MS: u64 = 200;

/// Slack on top of the pre-roll length before buffered audio counts as stale
/// (see [`PreRollBuffer::clear_if_stale`]).
pub const PRE_ROLL_STALE_SLACK_MS: u64 = 250;

/// Number of device-rate samples needed to hold `ms` of audio, clamped to
/// [`MAX_PRE_ROLL_MS`]. Returns 0 when pre-roll is disabled.
pub fn pre_roll_capacity_samples(sample_rate: u32, ms: u64) -> usize {
    let ms = ms.min(MAX_PRE_ROLL_MS);
    ((u64::from(sample_rate) * ms) / 1_000) as usize
}

/// Rolling window that keeps only the newest `capacity` mono samples.
#[derive(Debug, Default)]
pub struct PreRollBuffer {
    samples: VecDeque<f32>,
    capacity: usize,
    /// When audio last arrived. A stream can stall without an error (sleep,
    /// another app taking exclusive mode), leaving old audio in the window.
    last_push: Option<Instant>,
}

impl PreRollBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Resize the window. Shrinking keeps the newest samples; a capacity of 0
    /// disables pre-roll and releases the backing allocation.
    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity == self.capacity {
            return;
        }
        self.capacity = capacity;
        if capacity == 0 {
            self.samples = VecDeque::new();
            return;
        }
        let excess = self.samples.len().saturating_sub(capacity);
        self.samples.drain(..excess);
        if self.samples.capacity() < capacity {
            self.samples.reserve(capacity - self.samples.len());
        } else {
            self.samples.shrink_to(capacity);
        }
    }

    /// Append newly captured samples, evicting the oldest beyond `capacity`.
    pub fn push(&mut self, chunk: &[f32]) {
        if self.capacity == 0 || chunk.is_empty() {
            return;
        }
        if chunk.len() >= self.capacity {
            self.samples.clear();
            self.samples.extend(&chunk[chunk.len() - self.capacity..]);
            self.last_push = Some(Instant::now());
            return;
        }
        let overflow = (self.samples.len() + chunk.len()).saturating_sub(self.capacity);
        self.samples.drain(..overflow);
        self.samples.extend(chunk);
        self.last_push = Some(Instant::now());
    }

    /// Empty the window when its newest audio arrived more than `max_age`
    /// before `now`: after a silent device stall it would otherwise prepend
    /// audio that is minutes old.
    pub fn clear_if_stale(&mut self, now: Instant, max_age: Duration) {
        let stale = self
            .last_push
            .is_some_and(|pushed| now.saturating_duration_since(pushed) > max_age);
        if stale {
            self.samples.clear();
        }
    }

    /// Drop all but the newest `count` samples.
    pub fn keep_newest(&mut self, count: usize) {
        let excess = self.samples.len().saturating_sub(count);
        self.samples.drain(..excess);
    }

    /// Move the window, oldest first, into `out` (which is cleared first) and
    /// empty the buffer so the same audio can never be prepended twice.
    pub fn drain_into(&mut self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.samples.drain(..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contents(buffer: &mut PreRollBuffer) -> Vec<f32> {
        let mut out = Vec::new();
        buffer.drain_into(&mut out);
        out
    }

    #[test]
    fn capacity_scales_with_rate_and_clamps_to_max() {
        assert_eq!(pre_roll_capacity_samples(16_000, 0), 0);
        assert_eq!(pre_roll_capacity_samples(16_000, 300), 4_800);
        assert_eq!(pre_roll_capacity_samples(48_000, 300), 14_400);
        assert_eq!(pre_roll_capacity_samples(44_100, 1_000), 44_100);
        assert_eq!(pre_roll_capacity_samples(48_000, 5_000), 48_000);
    }

    #[test]
    fn stale_window_is_dropped() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(10);
        buffer.push(&[1.0, 2.0]);
        let pushed = Instant::now();
        let max_age = Duration::from_millis(550);

        buffer.clear_if_stale(pushed + Duration::from_millis(100), max_age);
        assert_eq!(buffer.len(), 2);
        buffer.clear_if_stale(pushed + Duration::from_secs(120), max_age);
        assert!(buffer.is_empty());
    }

    #[test]
    fn keep_newest_trims_the_oldest_samples() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(10);
        buffer.push(&[1.0, 2.0, 3.0, 4.0]);
        buffer.keep_newest(3);
        assert_eq!(contents(&mut buffer), vec![2.0, 3.0, 4.0]);
        buffer.push(&[5.0]);
        buffer.keep_newest(0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn disabled_buffer_stores_nothing() {
        let mut buffer = PreRollBuffer::new();
        buffer.push(&[1.0, 2.0, 3.0]);
        assert!(buffer.is_empty());
        assert!(contents(&mut buffer).is_empty());
    }

    #[test]
    fn keeps_only_the_newest_samples_in_order() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(4);
        buffer.push(&[1.0, 2.0]);
        buffer.push(&[3.0, 4.0, 5.0]);
        assert_eq!(buffer.len(), 4);
        buffer.push(&[6.0]);
        assert_eq!(contents(&mut buffer), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn oversized_chunk_keeps_its_tail() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(3);
        buffer.push(&[9.0]);
        buffer.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(contents(&mut buffer), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn drain_empties_the_window_so_audio_is_never_reused() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(8);
        buffer.push(&[1.0, 2.0, 3.0]);
        assert_eq!(contents(&mut buffer), vec![1.0, 2.0, 3.0]);
        assert!(buffer.is_empty());
        assert!(contents(&mut buffer).is_empty());
        // Still usable after draining.
        buffer.push(&[4.0]);
        assert_eq!(contents(&mut buffer), vec![4.0]);
    }

    #[test]
    fn shrinking_keeps_newest_and_zero_disables() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(6);
        buffer.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        buffer.set_capacity(2);
        assert_eq!(buffer.capacity(), 2);
        assert_eq!(buffer.len(), 2);
        buffer.set_capacity(0);
        assert!(buffer.is_empty());
        buffer.push(&[7.0]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn growing_preserves_existing_samples() {
        let mut buffer = PreRollBuffer::new();
        buffer.set_capacity(2);
        buffer.push(&[1.0, 2.0, 3.0]);
        buffer.set_capacity(4);
        buffer.push(&[4.0, 5.0]);
        assert_eq!(contents(&mut buffer), vec![2.0, 3.0, 4.0, 5.0]);
    }
}
