//! Pure logic for the replay bench: word-level disagreement, latency
//! statistics and recording selection. No I/O, so everything here is unit
//! tested directly.

use serde::{Deserialize, Serialize};
use specta::Type;

// Whether a word appears in both texts, only in the model output, or only in
// the saved reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum WordDiffKind {
    Same,
    Added,
    Removed,
}

// One word of an inline diff. `text` keeps the original spelling.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Type)]
pub struct WordDiffToken {
    pub kind: WordDiffKind,
    pub text: String,
}

/// Result of comparing a model's output against the saved reference text.
#[derive(Debug, Clone, PartialEq)]
pub struct WordDiff {
    /// Word-level edit distance (substitutions + insertions + deletions).
    pub distance: usize,
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub tokens: Vec<WordDiffToken>,
}

impl WordDiff {
    /// Share of words that differ, 0–100. The denominator is the longer of
    /// the two texts so the value stays bounded; two empty texts agree.
    pub fn differs_pct(&self) -> f64 {
        let longest = self.reference_words.max(self.hypothesis_words);
        if longest == 0 {
            0.0
        } else {
            self.distance as f64 * 100.0 / longest as f64
        }
    }
}

/// A word plus the key it is compared by (lowercase, edge punctuation
/// stripped), so "Hello," and "hello" agree.
struct Word<'a> {
    original: &'a str,
    key: String,
}

/// Chinese and Japanese are written without spaces, so each ideograph or
/// kana counts as one "word"; otherwise a CJK transcript is a single word
/// and differs% could only be 0 or 100. (Korean uses spaces.)
fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3000}'..='\u{303F}'   // CJK punctuation
            | '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
            | '\u{3400}'..='\u{4DBF}' // CJK Extension A
            | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
            | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
            | '\u{FF00}'..='\u{FFEF}' // Half/full-width forms
            | '\u{20000}'..='\u{2FA1F}' // Extensions B-F, supplement
    )
}

/// Split one whitespace-delimited chunk into comparison units: every CJK
/// character on its own, other characters in runs.
fn split_units(chunk: &str) -> impl Iterator<Item = &str> {
    let mut units = Vec::new();
    let mut run_start: Option<usize> = None;
    for (index, c) in chunk.char_indices() {
        if is_cjk(c) {
            if let Some(start) = run_start.take() {
                units.push(&chunk[start..index]);
            }
            units.push(&chunk[index..index + c.len_utf8()]);
        } else if run_start.is_none() {
            run_start = Some(index);
        }
    }
    if let Some(start) = run_start {
        units.push(&chunk[start..]);
    }
    units.into_iter()
}

fn words(text: &str) -> Vec<Word<'_>> {
    text.split_whitespace()
        .flat_map(split_units)
        .filter_map(|original| {
            let key = original
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            (!key.is_empty()).then_some(Word { original, key })
        })
        .collect()
}

/// Largest alignment table built for the inline diff (8 bytes per cell, so
/// about 16 MB). Longer pairs (e.g. a model stuck in a repetition loop) get
/// the edit distance from a two-row pass and no inline tokens.
const MAX_ALIGNMENT_CELLS: usize = 2_000_000;

/// Edit distance in O(m) memory, for pairs too large to align.
fn distance_only(reference: &[Word<'_>], hypothesis: &[Word<'_>]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (i, r) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = usize::from(r.key != h.key);
            current[j + 1] = (previous[j] + substitution)
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

/// Word-level Levenshtein alignment of `hypothesis` against `reference`.
pub fn word_diff(reference: &str, hypothesis: &str) -> WordDiff {
    let reference_words = words(reference);
    let hypothesis_words = words(hypothesis);
    let n = reference_words.len();
    let m = hypothesis_words.len();

    if (n + 1).saturating_mul(m + 1) > MAX_ALIGNMENT_CELLS {
        return WordDiff {
            distance: distance_only(&reference_words, &hypothesis_words),
            reference_words: n,
            hypothesis_words: m,
            tokens: Vec::new(),
        };
    }

    // dp[i][j] = distance between the first i reference and j hypothesis words.
    let width = m + 1;
    let mut dp = vec![0usize; (n + 1) * width];
    for i in 0..=n {
        dp[i * width] = i;
    }
    for (j, cell) in dp.iter_mut().enumerate().take(m + 1) {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let substitution =
                usize::from(reference_words[i - 1].key != hypothesis_words[j - 1].key);
            dp[i * width + j] = (dp[(i - 1) * width + (j - 1)] + substitution)
                .min(dp[(i - 1) * width + j] + 1)
                .min(dp[i * width + (j - 1)] + 1);
        }
    }

    // Walk back from the end to recover an alignment.
    let mut tokens = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let here = dp[i * width + j];
        if i > 0 && j > 0 {
            let same = reference_words[i - 1].key == hypothesis_words[j - 1].key;
            let diagonal = dp[(i - 1) * width + (j - 1)];
            if same && here == diagonal {
                tokens.push(token(WordDiffKind::Same, hypothesis_words[j - 1].original));
                i -= 1;
                j -= 1;
                continue;
            }
            if !same && here == diagonal + 1 {
                // Pushed in reverse, so the reference word reads first.
                tokens.push(token(WordDiffKind::Added, hypothesis_words[j - 1].original));
                tokens.push(token(
                    WordDiffKind::Removed,
                    reference_words[i - 1].original,
                ));
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && here == dp[(i - 1) * width + j] + 1 {
            tokens.push(token(
                WordDiffKind::Removed,
                reference_words[i - 1].original,
            ));
            i -= 1;
        } else {
            tokens.push(token(WordDiffKind::Added, hypothesis_words[j - 1].original));
            j -= 1;
        }
    }
    tokens.reverse();

    WordDiff {
        distance: dp[n * width + m],
        reference_words: n,
        hypothesis_words: m,
        tokens,
    }
}

fn token(kind: WordDiffKind, text: &str) -> WordDiffToken {
    WordDiffToken {
        kind,
        text: text.to_string(),
    }
}

/// Median of `values`; the mean of the two middle values for an even count.
pub fn median(values: &[f64]) -> Option<f64> {
    let sorted = sorted(values);
    let n = sorted.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(sorted[n / 2]),
        _ => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

/// Nearest-rank percentile (`p` in 0–100).
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    let sorted = sorted(values);
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p.clamp(0.0, 100.0) / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f64::total_cmp);
    sorted
}

/// Processing time divided by audio length; below 1.0 is faster than real
/// time. `None` when there is no audio to divide by.
pub fn real_time_factor(processing_ms: f64, audio_secs: f64) -> Option<f64> {
    (audio_secs > 0.0).then(|| processing_ms / 1000.0 / audio_secs)
}

// Which saved recordings to replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayBenchSelection {
    Recent { count: u32 },
    Starred,
}

/// Upper bound on recordings in one run, whatever the selection.
pub const MAX_BENCH_ENTRIES: usize = 100;

/// The parts of a history entry that selection looks at.
#[derive(Debug, Clone)]
pub struct Candidate<'a> {
    pub id: i64,
    pub timestamp: i64,
    pub saved: bool,
    pub transcription_text: &'a str,
}

/// Pick the recordings to replay, newest first. Entries without a WAV on disk
/// or without saved text (nothing to compare against) are skipped, and only
/// then is the count applied, so "last 10" means ten usable recordings.
///
/// `has_audio` is only asked about entries that pass every cheap check, newest
/// first, and stops being asked once the selection is full, so a count over a
/// long History stats a handful of WAVs instead of all of them.
pub fn select_entries(
    candidates: &[Candidate<'_>],
    selection: ReplayBenchSelection,
    mut has_audio: impl FnMut(&Candidate<'_>) -> bool,
) -> Vec<i64> {
    let mut usable: Vec<&Candidate<'_>> = candidates
        .iter()
        .filter(|c| !c.transcription_text.trim().is_empty())
        .filter(|c| match selection {
            ReplayBenchSelection::Recent { .. } => true,
            ReplayBenchSelection::Starred => c.saved,
        })
        .collect();
    usable.sort_by(|a, b| b.timestamp.cmp(&a.timestamp).then(b.id.cmp(&a.id)));
    let limit = match selection {
        ReplayBenchSelection::Recent { count } => (count as usize).min(MAX_BENCH_ENTRIES),
        ReplayBenchSelection::Starred => MAX_BENCH_ENTRIES,
    };
    usable
        .into_iter()
        .filter(|c| has_audio(c))
        .take(limit)
        .map(|c| c.id)
        .collect()
}

/// The text a run is compared against. With cleanup on, the polished text is
/// the like-for-like reference when one was saved; otherwise the raw
/// transcription is.
pub fn reference_text<'a>(
    transcription_text: &'a str,
    post_processed_text: Option<&'a str>,
    include_cleanup: bool,
) -> &'a str {
    match post_processed_text {
        Some(polished) if include_cleanup && !polished.trim().is_empty() => polished,
        _ => transcription_text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(diff: &WordDiff) -> Vec<(WordDiffKind, &str)> {
        diff.tokens
            .iter()
            .map(|t| (t.kind, t.text.as_str()))
            .collect()
    }

    #[test]
    fn cjk_text_is_compared_per_character() {
        // One wrong character out of five, not a 100% mismatch.
        let diff = word_diff("今日は晴れ", "今日は雨れ");
        assert_eq!(diff.reference_words, 5);
        assert_eq!(diff.distance, 1);
        assert!((diff.differs_pct() - 20.0).abs() < 1e-9);
        // Mixed scripts keep Latin words whole; CJK punctuation is ignored.
        let diff = word_diff("我用 Handy 写作。", "我用 handy 写作");
        assert_eq!(diff.distance, 0);
        assert_eq!(diff.reference_words, 5);
    }

    #[test]
    fn huge_texts_get_distance_without_alignment() {
        let reference = "word ".repeat(1_500);
        let hypothesis = format!("{}extra", "word ".repeat(1_500));
        let diff = word_diff(&reference, &hypothesis);
        assert_eq!(diff.distance, 1);
        assert!(diff.tokens.is_empty());
        // The O(m) pass agrees with the full table on small inputs.
        let r = words("the quick brown fox");
        let h = words("a quick brown dog jumps");
        assert_eq!(
            distance_only(&r, &h),
            word_diff("the quick brown fox", "a quick brown dog jumps").distance
        );
    }

    #[test]
    fn identical_texts_do_not_differ() {
        let diff = word_diff("The quick brown fox", "The quick brown fox");
        assert_eq!(diff.distance, 0);
        assert_eq!(diff.differs_pct(), 0.0);
        assert!(diff.tokens.iter().all(|t| t.kind == WordDiffKind::Same));
    }

    #[test]
    fn case_and_edge_punctuation_are_ignored() {
        let diff = word_diff("Hello, world.", "hello world");
        assert_eq!(diff.distance, 0);
        assert_eq!(diff.differs_pct(), 0.0);
    }

    #[test]
    fn substitution_counts_once() {
        let diff = word_diff("send the report today", "send the memo today");
        assert_eq!(diff.distance, 1);
        assert_eq!(diff.differs_pct(), 25.0);
        assert_eq!(
            kinds(&diff),
            vec![
                (WordDiffKind::Same, "send"),
                (WordDiffKind::Same, "the"),
                (WordDiffKind::Removed, "report"),
                (WordDiffKind::Added, "memo"),
                (WordDiffKind::Same, "today"),
            ]
        );
    }

    #[test]
    fn insertion_and_deletion_are_aligned() {
        let inserted = word_diff("call me later", "call me much later");
        assert_eq!(inserted.distance, 1);
        assert_eq!(
            kinds(&inserted),
            vec![
                (WordDiffKind::Same, "call"),
                (WordDiffKind::Same, "me"),
                (WordDiffKind::Added, "much"),
                (WordDiffKind::Same, "later"),
            ]
        );
        // Denominator is the longer text (4 words).
        assert_eq!(inserted.differs_pct(), 25.0);

        let deleted = word_diff("um call me later", "call me later");
        assert_eq!(deleted.distance, 1);
        assert_eq!(deleted.tokens[0].kind, WordDiffKind::Removed);
        assert_eq!(deleted.tokens[0].text, "um");
    }

    #[test]
    fn empty_texts_are_bounded() {
        assert_eq!(word_diff("", "").differs_pct(), 0.0);
        assert_eq!(word_diff("", "something new").differs_pct(), 100.0);
        assert_eq!(word_diff("all gone", "").differs_pct(), 100.0);
        assert_eq!(word_diff("a b c", "x y z w").differs_pct(), 100.0);
    }

    #[test]
    fn pure_punctuation_tokens_are_skipped() {
        let diff = word_diff("yes — exactly", "yes exactly");
        assert_eq!(diff.distance, 0);
        assert_eq!(diff.reference_words, 2);
    }

    #[test]
    fn median_and_percentile() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(percentile(&values, 90.0), Some(9.0));
        assert_eq!(percentile(&values, 100.0), Some(10.0));
        assert_eq!(percentile(&[7.0], 90.0), Some(7.0));
        assert_eq!(percentile(&[], 90.0), None);
        assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn real_time_factor_handles_empty_audio() {
        assert_eq!(real_time_factor(500.0, 2.0), Some(0.25));
        assert_eq!(real_time_factor(500.0, 0.0), None);
    }

    fn candidate(id: i64, timestamp: i64, saved: bool, text: &str) -> Candidate<'_> {
        Candidate {
            id,
            timestamp,
            saved,
            transcription_text: text,
        }
    }

    /// Audio check that reports `missing` ids as deleted and records every id
    /// it was asked about.
    fn audio_except<'m>(
        missing: &'m [i64],
        asked: &'m std::cell::RefCell<Vec<i64>>,
    ) -> impl FnMut(&Candidate<'_>) -> bool + 'm {
        move |c| {
            asked.borrow_mut().push(c.id);
            !missing.contains(&c.id)
        }
    }

    #[test]
    fn recent_selection_skips_missing_audio_before_counting() {
        let candidates = vec![
            candidate(1, 100, false, "oldest"),
            candidate(2, 200, false, "middle"),
            candidate(3, 300, false, "audio deleted"),
            candidate(4, 400, false, "newest"),
            candidate(5, 500, false, "   "),
        ];
        let asked = std::cell::RefCell::new(Vec::new());
        let picked = select_entries(
            &candidates,
            ReplayBenchSelection::Recent { count: 2 },
            audio_except(&[3], &asked),
        );
        assert_eq!(picked, vec![4, 2]);
        // Blank entries are never probed, and probing stops once full.
        assert_eq!(*asked.borrow(), vec![4, 3, 2]);
    }

    #[test]
    fn recent_selection_returns_fewer_when_not_enough() {
        let asked = std::cell::RefCell::new(Vec::new());
        let candidates = vec![candidate(1, 100, false, "only one")];
        let picked = select_entries(
            &candidates,
            ReplayBenchSelection::Recent { count: 10 },
            audio_except(&[], &asked),
        );
        assert_eq!(picked, vec![1]);
        assert!(select_entries(
            &[],
            ReplayBenchSelection::Recent { count: 10 },
            audio_except(&[], &asked)
        )
        .is_empty());
    }

    #[test]
    fn same_second_entries_order_by_id() {
        let asked = std::cell::RefCell::new(Vec::new());
        let candidates = vec![
            candidate(1, 100, false, "first"),
            candidate(2, 100, false, "second"),
        ];
        let picked = select_entries(
            &candidates,
            ReplayBenchSelection::Recent { count: 1 },
            audio_except(&[], &asked),
        );
        assert_eq!(picked, vec![2]);
    }

    #[test]
    fn starred_selection_only_takes_saved_entries() {
        let candidates = vec![
            candidate(1, 100, true, "starred old"),
            candidate(2, 200, false, "not starred"),
            candidate(3, 300, true, "starred but missing"),
            candidate(4, 400, true, "starred new"),
        ];
        let asked = std::cell::RefCell::new(Vec::new());
        let picked = select_entries(
            &candidates,
            ReplayBenchSelection::Starred,
            audio_except(&[3], &asked),
        );
        assert_eq!(picked, vec![4, 1]);
        // Unstarred entries are never probed.
        assert!(!asked.borrow().contains(&2));
    }

    #[test]
    fn reference_prefers_polished_only_with_cleanup() {
        assert_eq!(reference_text("raw", Some("Polished."), true), "Polished.");
        assert_eq!(reference_text("raw", Some("Polished."), false), "raw");
        assert_eq!(reference_text("raw", None, true), "raw");
        assert_eq!(reference_text("raw", Some("  "), true), "raw");
    }
}
