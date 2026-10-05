import assert from "node:assert/strict";
import { entryMatchesQuery, getEntryTexts, splitByQuery } from "./historyText";

const entry = (
  raw: string,
  polished: string | null,
  requested = polished !== null,
) => ({
  transcription_text: raw,
  post_processed_text: polished,
  post_process_requested: requested,
});

// Polished text is primary; the original is offered when it differs.
{
  const texts = getEntryTexts(entry("um hello there", "Hello there."));
  assert.equal(texts.primaryText, "Hello there.");
  assert.equal(texts.originalText, "um hello there");
  assert.equal(texts.hasDistinctOriginal, true);
  assert.equal(texts.status, "cleanedUp");
}

// Identical polished text hides the disclosure but still counts as cleaned up.
{
  const texts = getEntryTexts(entry("Hello.", " Hello. "));
  assert.equal(texts.hasDistinctOriginal, false);
  assert.equal(texts.status, "cleanedUp");
}

// No cleanup requested: the raw text, badge "original".
{
  const texts = getEntryTexts(entry("raw words", null, false));
  assert.equal(texts.primaryText, "raw words");
  assert.equal(texts.hasDistinctOriginal, false);
  assert.equal(texts.status, "original");
}

// Cleanup requested but nothing came back (or only whitespace): failed.
assert.equal(getEntryTexts(entry("raw", null, true)).status, "cleanupFailed");
assert.equal(getEntryTexts(entry("raw", "  ", true)).status, "cleanupFailed");
// Requested but never sent (cleanup misconfigured): not a failure.
assert.equal(
  getEntryTexts({ ...entry("raw", null, true), cleanup_attempted: false })
    .status,
  "original",
);
assert.equal(
  getEntryTexts({ ...entry("raw", null, true), cleanup_attempted: true })
    .status,
  "cleanupFailed",
);
assert.equal(getEntryTexts(entry("raw", "  ", true)).primaryText, "raw");

// Cleanup missed its time limit and is still running: "cleaning up", which
// wins over "cleanup failed" (requested, no polished text yet).
{
  const texts = getEntryTexts({
    ...entry("raw words", null, true),
    cleanup_state: "pending",
  });
  assert.equal(texts.status, "cleaningUp");
  assert.equal(texts.primaryText, "raw words");
}

// Polished text arrived after the time limit: "cleaned up late".
{
  const texts = getEntryTexts({
    ...entry("um raw", "Raw.", true),
    cleanup_state: "late",
  });
  assert.equal(texts.status, "cleanedUpLate");
  assert.equal(texts.primaryText, "Raw.");
  assert.equal(texts.hasDistinctOriginal, true);
}

// "late" without polished text (a late request that then failed is reset to
// null by the backend, but be defensive): treated as failed.
assert.equal(
  getEntryTexts({ ...entry("raw", null, true), cleanup_state: "late" }).status,
  "cleanupFailed",
);

// A failed late request goes back to null: the existing "cleanup failed".
assert.equal(
  getEntryTexts({ ...entry("raw", null, true), cleanup_state: null }).status,
  "cleanupFailed",
);

// Failed transcription: no badge.
assert.equal(getEntryTexts(entry("", null, true)).status, null);

// Highlight splitting is case-insensitive and treats regex characters literally.
assert.deepEqual(splitByQuery("Budget and budget", "BUDGET"), [
  { text: "Budget", match: true },
  { text: " and ", match: false },
  { text: "budget", match: true },
]);
assert.deepEqual(splitByQuery("cost (approx.) 5$", "(approx.)"), [
  { text: "cost ", match: false },
  { text: "(approx.)", match: true },
  { text: " 5$", match: false },
]);
assert.deepEqual(splitByQuery("text", "  "), [{ text: "text", match: false }]);

// Live-entry matching mirrors the backend: raw or polished, case-insensitive.
assert.equal(entryMatchesQuery(entry("raw words", "Polished"), "POLISH"), true);
assert.equal(entryMatchesQuery(entry("raw words", null), "words"), true);
assert.equal(entryMatchesQuery(entry("raw words", null), "missing"), false);
assert.equal(entryMatchesQuery(entry("anything", null), ""), true);

console.log("historyText tests passed");
