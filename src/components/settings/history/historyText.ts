import type { HistoryEntry } from "@/bindings";

/**
 * Cleanup state shown as a badge on a history entry.
 * - cleanedUp: post-processing produced polished text
 * - cleanedUpLate: polished text arrived after the cleanup time limit
 *   (the original was pasted)
 * - cleaningUp: the cleanup missed its time limit and is still running
 * - cleanupFailed: post-processing was requested but produced nothing
 * - original: no post-processing ran
 */
export type EntryStatus =
  | "cleanedUp"
  | "cleanedUpLate"
  | "cleaningUp"
  | "original"
  | "cleanupFailed";

export interface EntryTexts {
  /** What the entry is used as: polished text when present, else the raw text. */
  primaryText: string;
  /** The raw transcription. */
  originalText: string;
  /** True when polished text exists and differs from the raw transcription. */
  hasDistinctOriginal: boolean;
  /** Badge to show, or null when there is no transcription at all. */
  status: EntryStatus | null;
}

type EntryTextFields = Pick<
  HistoryEntry,
  "transcription_text" | "post_processed_text" | "post_process_requested"
> &
  Partial<Pick<HistoryEntry, "cleanup_state">>;

export const getEntryTexts = (entry: EntryTextFields): EntryTexts => {
  const originalText = entry.transcription_text;
  const polished =
    entry.post_processed_text && entry.post_processed_text.trim().length > 0
      ? entry.post_processed_text
      : null;
  const primaryText = polished ?? originalText;
  const hasTranscription = originalText.trim().length > 0;

  let status: EntryStatus | null = null;
  if (entry.cleanup_state === "pending" && hasTranscription) {
    // A late cleanup is still running; this wins over "cleanup failed".
    status = "cleaningUp";
  } else if (polished !== null) {
    status = entry.cleanup_state === "late" ? "cleanedUpLate" : "cleanedUp";
  } else if (hasTranscription) {
    status = entry.post_process_requested ? "cleanupFailed" : "original";
  }

  return {
    primaryText,
    originalText,
    hasDistinctOriginal:
      polished !== null && polished.trim() !== originalText.trim(),
    status,
  };
};

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export interface TextPart {
  text: string;
  match: boolean;
}

/** Splits `text` into alternating non-matching / matching parts (case-insensitive). */
export const splitByQuery = (text: string, query: string): TextPart[] => {
  const needle = query.trim();
  if (!needle) return [{ text, match: false }];
  const parts = text.split(new RegExp(`(${escapeRegExp(needle)})`, "gi"));
  // With one capture group, odd indices are the matches.
  return parts
    .map((part, index) => ({ text: part, match: index % 2 === 1 }))
    .filter((part) => part.text.length > 0);
};

/** Client-side mirror of the backend search, for entries that arrive live. */
export const entryMatchesQuery = (
  entry: EntryTextFields,
  query: string,
): boolean => {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [entry.transcription_text, entry.post_processed_text ?? ""].some(
    (text) => text.toLowerCase().includes(needle),
  );
};
