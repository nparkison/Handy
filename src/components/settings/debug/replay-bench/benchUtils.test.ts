import assert from "node:assert/strict";
import type {
  ReplayBenchEntry,
  ReplayBenchModel,
  ReplayBenchResult,
} from "@/bindings";
import {
  buildCsv,
  formatMs,
  formatPct,
  formatRtf,
  resultKey,
  sortByDisagreement,
  type BenchResults,
} from "./benchUtils";

const entry = (entry_id: number, reference_text = "hi"): ReplayBenchEntry => ({
  entry_id,
  timestamp: entry_id * 10,
  reference_text,
  duration_secs: 1,
  readable: true,
});

const models: ReplayBenchModel[] = [
  { model_id: "a", model_name: "Model A" },
  { model_id: "b", model_name: "Model B" },
];

const result = (
  entry_id: number,
  model_id: string,
  differs_pct: number | null,
  failure: ReplayBenchResult["failure"] = null,
): ReplayBenchResult => ({
  entry_id,
  model_id,
  failure,
  text: "text, with comma",
  cleanup_text: null,
  cleanup_failed: false,
  differs_pct,
  diff: [],
  transcribe_ms: 120,
  rtf: 0.05,
  cleanup_ms: null,
});

const results: BenchResults = {
  [resultKey(1, "a")]: result(1, "a", 10),
  [resultKey(1, "b")]: result(1, "b", 40),
  [resultKey(2, "a")]: result(2, "a", 0),
  [resultKey(3, "a")]: result(3, "a", null, "decode_failed"),
};

const sorted = sortByDisagreement(
  [entry(1), entry(2), entry(3), entry(4)],
  models,
  results,
);
assert.deepEqual(
  sorted.map((e) => e.entry_id),
  [3, 1, 2, 4],
);

assert.equal(formatMs(null), "–");
assert.equal(formatMs(850.4), "850 ms");
assert.equal(formatMs(1234), "1.23 s");
assert.equal(formatPct(12.6), "13%");
assert.equal(formatRtf(0.0123), "0.012");
assert.equal(formatRtf(0.5), "0.50");

const csv = buildCsv([entry(1, 'say "hi"')], models, results, {});
const lines = csv.split("\n");
assert.equal(lines.length, 3);
assert.ok(lines[1].includes('"say ""hi"""'));
assert.ok(lines[1].includes('"text, with comma"'));
assert.ok(lines[1].startsWith("1,10,a,Model A,"));

console.log("replay bench utils: all assertions passed");
