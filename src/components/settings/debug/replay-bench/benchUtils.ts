import type {
  ReplayBenchEntry,
  ReplayBenchModel,
  ReplayBenchModelSummary,
  ReplayBenchResult,
} from "@/bindings";

export type BenchResults = Record<string, ReplayBenchResult>;

export const resultKey = (entryId: number, modelId: string): string =>
  `${entryId}:${modelId}`;

/** A failed clip sorts above any disagreement so problems surface first. */
const FAILURE_SCORE = 101;

/** Highest disagreement for a recording across models (-1 if none yet). */
export const rowDisagreement = (
  entryId: number,
  models: ReplayBenchModel[],
  results: BenchResults,
): number => {
  let worst = -1;
  for (const model of models) {
    const result = results[resultKey(entryId, model.model_id)];
    if (!result) continue;
    const score =
      result.failure !== null ? FAILURE_SCORE : (result.differs_pct ?? -1);
    worst = Math.max(worst, score);
  }
  return worst;
};

/** Most disagreement first; ties keep newest-first order. */
export const sortByDisagreement = (
  entries: ReplayBenchEntry[],
  models: ReplayBenchModel[],
  results: BenchResults,
): ReplayBenchEntry[] =>
  entries
    .map((entry, index) => ({
      entry,
      index,
      score: rowDisagreement(entry.entry_id, models, results),
    }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(({ entry }) => entry);

export const formatMs = (ms: number | null | undefined): string => {
  if (ms === null || ms === undefined) return "–";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
};

export const formatPct = (pct: number | null | undefined): string =>
  pct === null || pct === undefined ? "–" : `${Math.round(pct)}%`;

export const formatRtf = (rtf: number | null | undefined): string =>
  rtf === null || rtf === undefined ? "–" : rtf.toFixed(rtf < 0.1 ? 3 : 2);

const csvCell = (value: string | number | null | undefined): string => {
  if (value === null || value === undefined) return "";
  let text = String(value);
  // Dictated text that starts like a formula would run as one when the CSV
  // is opened in a spreadsheet; a leading quote makes it plain text.
  if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) {
    text = `'${text}`;
  }
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const CSV_HEADER = [
  "entry_id",
  "timestamp",
  "model_id",
  "model_name",
  "reference_text",
  "model_text",
  "cleanup_text",
  "differs_pct",
  "load_ms",
  "transcribe_ms",
  "rtf",
  "cleanup_ms",
  "failure",
];

/** Long format: one line per recording and model. */
export const buildCsv = (
  entries: ReplayBenchEntry[],
  models: ReplayBenchModel[],
  results: BenchResults,
  summaries: Record<string, ReplayBenchModelSummary>,
): string => {
  const lines = [CSV_HEADER.join(",")];
  for (const entry of entries) {
    for (const model of models) {
      const result = results[resultKey(entry.entry_id, model.model_id)];
      if (!result) continue;
      lines.push(
        [
          entry.entry_id,
          entry.timestamp,
          model.model_id,
          model.model_name,
          entry.reference_text,
          result.text,
          result.cleanup_text,
          result.differs_pct === null ? null : result.differs_pct.toFixed(1),
          summaries[model.model_id]?.load_ms?.toFixed(0),
          result.transcribe_ms?.toFixed(0),
          result.rtf?.toFixed(4),
          result.cleanup_ms?.toFixed(0),
          result.failure,
        ]
          .map(csvCell)
          .join(","),
      );
    }
  }
  return lines.join("\n");
};
