import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  commands,
  type ReplayBenchEntry,
  type ReplayBenchModel,
  type ReplayBenchResult,
  type ReplayBenchSelection,
  type WordDiffToken,
} from "@/bindings";
import { SettingsGroup } from "../../../ui/SettingsGroup";
import { SettingContainer } from "../../../ui/SettingContainer";
import { Dropdown } from "../../../ui/Dropdown";
import { ToggleSwitch } from "../../../ui/ToggleSwitch";
import { Button } from "../../../ui/Button";
import { useModelStore } from "../../../../stores/modelStore";
import { useReplayBenchStore } from "../../../../stores/replayBenchStore";
import { copyToClipboard } from "../../history/clipboard";
import {
  buildCsv,
  formatMs,
  formatPct,
  formatRtf,
  resultKey,
  sortByDisagreement,
} from "./benchUtils";

const SELECTION_OPTIONS = ["recent_10", "recent_20", "recent_50", "starred"];
const DEFAULT_SELECTION = "recent_10";

const toSelection = (value: string): ReplayBenchSelection => {
  if (value === "starred") return { kind: "starred" };
  const count = Number(value.replace("recent_", ""));
  return { kind: "recent", count: Number.isFinite(count) ? count : 10 };
};

interface ReplayBenchProps {
  onOpenModels?: () => void;
}

export const ReplayBench: React.FC<ReplayBenchProps> = ({ onOpenModels }) => {
  const { t } = useTranslation();
  const allModels = useModelStore((s) => s.models);
  const currentModel = useModelStore((s) => s.currentModel);
  const bench = useReplayBenchStore();

  const [selectionValue, setSelectionValue] = useState(DEFAULT_SELECTION);
  const [includeCleanup, setIncludeCleanup] = useState(false);
  const [checkedModels, setCheckedModels] = useState<string[] | null>(null);
  const [available, setAvailable] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);

  const installed = useMemo(
    () => allModels.filter((m) => m.is_downloaded),
    [allModels],
  );
  const busy = bench.status === "running" || bench.status === "starting";

  const initializeBench = useReplayBenchStore((s) => s.initialize);
  useEffect(() => {
    void initializeBench();
  }, [initializeBench]);

  // Pre-check the active model plus one other, once models are known.
  useEffect(() => {
    if (checkedModels !== null || installed.length === 0) return;
    const active = installed.find((m) => m.id === currentModel);
    const other = installed.find((m) => m.id !== currentModel);
    setCheckedModels(
      [active?.id, other?.id].filter((id): id is string => Boolean(id)),
    );
  }, [checkedModels, installed, currentModel]);

  // How many recordings the current selection would replay.
  useEffect(() => {
    let cancelled = false;
    commands
      .countReplayBenchRecordings(toSelection(selectionValue))
      .then((result) => {
        if (!cancelled) {
          setAvailable(result.status === "ok" ? result.data : null);
        }
      })
      .catch(() => {
        if (!cancelled) setAvailable(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectionValue, bench.status]);

  const selectedModels = (checkedModels ?? []).filter((id) =>
    installed.some((m) => m.id === id),
  );
  const tooFewInstalled = installed.length < 2;
  const noRecordings = available === 0;
  const canRun =
    !busy && !tooFewInstalled && !noRecordings && selectedModels.length > 0;

  const toggleModel = (id: string) => {
    setCheckedModels((prev) => {
      const current = prev ?? [];
      return current.includes(id)
        ? current.filter((m) => m !== id)
        : [...current, id];
    });
  };

  const run = () => {
    // Keep the user's ordering stable: installed-list order.
    const modelIds = installed
      .filter((m) => selectedModels.includes(m.id))
      .map((m) => m.id);
    void bench.start({
      selection: toSelection(selectionValue),
      model_ids: modelIds,
      include_cleanup: includeCleanup,
    });
  };

  const copyCsv = async () => {
    const ok = await copyToClipboard(
      buildCsv(bench.entries, bench.models, bench.results, bench.summaries),
    );
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  const selectionOptions = SELECTION_OPTIONS.map((value) => {
    const selection = toSelection(value);
    return {
      value,
      label:
        selection.kind === "recent"
          ? t("settings.debug.replayBench.recordings.last", {
              count: selection.count,
            })
          : t("settings.debug.replayBench.recordings.starred"),
    };
  });

  const errorMessage = (error: string): string => {
    switch (error) {
      case "no_recordings":
        return t("settings.debug.replayBench.empty.noRecordings");
      case "already_running":
        return t("settings.debug.replayBench.errors.alreadyRunning");
      case "no_models":
        return t("settings.debug.replayBench.errors.noModels");
      default:
        return t("settings.debug.replayBench.errors.failed", { error });
    }
  };

  const progressLine = (): string | null => {
    const p = bench.progress;
    if (bench.status === "starting") {
      return t("settings.debug.replayBench.progress.starting");
    }
    if (!p) return null;
    switch (p.phase) {
      case "loading":
        return t("settings.debug.replayBench.progress.loading", {
          model: p.modelName,
        });
      case "paused":
        return t("settings.debug.replayBench.progress.paused");
      case "cleanup":
        return t("settings.debug.replayBench.progress.cleanup", {
          index: p.entryIndex,
          total: p.entryTotal,
          model: p.modelName,
        });
      default:
        return t("settings.debug.replayBench.progress.transcribing", {
          index: p.entryIndex,
          total: p.entryTotal,
          model: p.modelName,
        });
    }
  };

  const progressText = progressLine();
  const hasResults = bench.entries.length > 0 && bench.models.length > 0;

  return (
    <SettingsGroup
      title={t("settings.debug.replayBench.title")}
      description={t("settings.debug.replayBench.description")}
    >
      <SettingContainer
        title={t("settings.debug.replayBench.recordings.label")}
        description={t("settings.debug.replayBench.recordings.description")}
        descriptionMode="tooltip"
        grouped={true}
        disabled={busy}
      >
        <Dropdown
          options={selectionOptions}
          selectedValue={selectionValue}
          onSelect={setSelectionValue}
          disabled={busy}
        />
      </SettingContainer>
      <SettingContainer
        title={t("settings.debug.replayBench.models.label")}
        description={t("settings.debug.replayBench.models.description")}
        descriptionMode="tooltip"
        grouped={true}
        layout="stacked"
        disabled={busy}
      >
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {installed.map((model) => (
            <label
              key={model.id}
              className="flex items-center gap-2 text-sm cursor-pointer"
            >
              <input
                type="checkbox"
                checked={selectedModels.includes(model.id)}
                onChange={() => toggleModel(model.id)}
                disabled={busy}
                className="accent-logo-primary"
              />
              <span>
                {model.name}
                {model.id === currentModel && (
                  <span className="text-mid-gray">
                    {" "}
                    {t("settings.debug.replayBench.models.active")}
                  </span>
                )}
              </span>
            </label>
          ))}
        </div>
      </SettingContainer>
      <ToggleSwitch
        checked={includeCleanup}
        onChange={setIncludeCleanup}
        disabled={busy}
        label={t("settings.debug.replayBench.cleanup.label")}
        description={t("settings.debug.replayBench.cleanup.description")}
        descriptionMode="tooltip"
        grouped={true}
      />
      <div className="p-4 space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Button size="sm" onClick={run} disabled={!canRun}>
            {t("settings.debug.replayBench.run")}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void bench.stop()}
            disabled={bench.status !== "running"}
          >
            {t("settings.debug.replayBench.stop")}
          </Button>
          {available !== null && available > 0 && !busy && (
            <span className="text-xs text-mid-gray">
              {t("settings.debug.replayBench.recordings.available", {
                count: available,
              })}
            </span>
          )}
        </div>
        <p className="text-xs text-mid-gray">
          {t("settings.debug.replayBench.loadNote")}
        </p>
        {tooFewInstalled && (
          <p className="text-sm">
            {t("settings.debug.replayBench.oneModel")}{" "}
            {onOpenModels && (
              <button
                type="button"
                onClick={onOpenModels}
                className="underline text-logo-primary cursor-pointer"
              >
                {t("settings.debug.replayBench.openModels")}
              </button>
            )}
          </p>
        )}
        {noRecordings && (
          <p className="text-sm">
            {selectionValue === "starred"
              ? t("settings.debug.replayBench.empty.noStarred")
              : t("settings.debug.replayBench.empty.noRecordings")}
          </p>
        )}
        <div aria-live="polite" className="text-sm">
          {progressText && <p>{progressText}</p>}
          {bench.status === "done" && (
            <p>{t("settings.debug.replayBench.finished")}</p>
          )}
          {bench.status === "stopped" && (
            <p>{t("settings.debug.replayBench.stopped")}</p>
          )}
          {bench.error && (
            <p className="text-red-500">{errorMessage(bench.error)}</p>
          )}
        </div>
      </div>
      {hasResults && (
        <div className="p-4 space-y-4">
          <SummaryTable />
          <ResultsTable />
          <div className="flex items-center gap-2">
            <Button size="sm" variant="secondary" onClick={copyCsv}>
              {copied
                ? t("settings.debug.replayBench.copied")
                : t("settings.debug.replayBench.copyCsv")}
            </Button>
            <span className="text-xs text-mid-gray">
              {t("settings.debug.replayBench.notSaved")}
            </span>
          </div>
        </div>
      )}
    </SettingsGroup>
  );
};

const thClass =
  "px-2 py-1 text-left font-medium text-mid-gray whitespace-nowrap";
const tdClass = "px-2 py-1 align-top whitespace-nowrap";

const SummaryTable: React.FC = () => {
  const { t } = useTranslation();
  const { models, summaries, includeCleanup } = useReplayBenchStore();

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <caption className="text-left text-xs font-medium pb-1">
          {t("settings.debug.replayBench.table.summary")}
        </caption>
        <thead>
          <tr className="border-b border-mid-gray/20">
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.model")}
            </th>
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.meanDiffers")}
            </th>
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.load")}
            </th>
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.transcribeMedianP90")}
            </th>
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.rtfMedian")}
            </th>
            {includeCleanup && (
              <th scope="col" className={thClass}>
                {t("settings.debug.replayBench.table.cleanupMedianP90")}
              </th>
            )}
            <th scope="col" className={thClass}>
              {t("settings.debug.replayBench.table.failed")}
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const s = summaries[model.model_id];
            return (
              <tr key={model.model_id} className="border-b border-mid-gray/10">
                <th scope="row" className={`${tdClass} text-left font-medium`}>
                  {model.model_name}
                </th>
                {s?.load_error ? (
                  <td
                    className={`${tdClass} text-red-500 whitespace-normal`}
                    colSpan={includeCleanup ? 6 : 5}
                  >
                    {t("settings.debug.replayBench.loadFailed", {
                      error: s.load_error,
                    })}
                  </td>
                ) : (
                  <>
                    <td className={tdClass}>
                      {formatPct(s?.mean_differs_pct)}
                    </td>
                    <td className={tdClass}>{formatMs(s?.load_ms)}</td>
                    <td className={tdClass}>
                      {`${formatMs(s?.transcribe_median_ms)} / ${formatMs(s?.transcribe_p90_ms)}`}
                    </td>
                    <td className={tdClass}>{formatRtf(s?.rtf_median)}</td>
                    {includeCleanup && (
                      <td className={tdClass}>
                        {`${formatMs(s?.cleanup_median_ms)} / ${formatMs(s?.cleanup_p90_ms)}`}
                      </td>
                    )}
                    <td className={tdClass}>{s ? String(s.failed) : ""}</td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-xs text-mid-gray pt-1">
        {t("settings.debug.replayBench.table.differsHint")}{" "}
        {t("settings.debug.replayBench.table.rtfHint")}
      </p>
    </div>
  );
};

const ResultsTable: React.FC = () => {
  const { t } = useTranslation();
  const { entries, models, results, includeCleanup } = useReplayBenchStore();
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const rows = useMemo(
    () => sortByDisagreement(entries, models, results),
    [entries, models, results],
  );
  const perModelColumns = includeCleanup ? 4 : 3;
  const totalColumns = 1 + models.length * perModelColumns;

  const toggle = (id: number) =>
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <caption className="text-left text-xs font-medium pb-1">
          {t("settings.debug.replayBench.table.recordings")}
        </caption>
        <thead>
          <tr>
            <th scope="col" rowSpan={2} className={`${thClass} align-bottom`}>
              {t("settings.debug.replayBench.table.savedText")}
            </th>
            {models.map((model) => (
              <th
                key={model.model_id}
                scope="colgroup"
                colSpan={perModelColumns}
                className={`${thClass} border-l border-mid-gray/20`}
              >
                {model.model_name}
              </th>
            ))}
          </tr>
          <tr className="border-b border-mid-gray/20">
            {models.map((model) => (
              <React.Fragment key={model.model_id}>
                <th
                  scope="col"
                  className={`${thClass} border-l border-mid-gray/20`}
                >
                  {t("settings.debug.replayBench.table.differs")}
                </th>
                <th scope="col" className={thClass}>
                  {t("settings.debug.replayBench.table.transcribe")}
                </th>
                <th scope="col" className={thClass}>
                  {t("settings.debug.replayBench.table.rtf")}
                </th>
                {includeCleanup && (
                  <th scope="col" className={thClass}>
                    {t("settings.debug.replayBench.table.cleanup")}
                  </th>
                )}
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((entry) => (
            <React.Fragment key={entry.entry_id}>
              <tr className="border-b border-mid-gray/10">
                <td className="px-2 py-1 align-top min-w-48 max-w-72">
                  <p className="line-clamp-2 select-text">
                    {entry.reference_text}
                  </p>
                  <button
                    type="button"
                    onClick={() => toggle(entry.entry_id)}
                    aria-expanded={Boolean(expanded[entry.entry_id])}
                    className="text-logo-primary underline cursor-pointer"
                  >
                    {expanded[entry.entry_id]
                      ? t("settings.debug.replayBench.table.hideDiff")
                      : t("settings.debug.replayBench.table.showDiff")}
                  </button>
                </td>
                {models.map((model) => (
                  <ResultCells
                    key={model.model_id}
                    entry={entry}
                    result={results[resultKey(entry.entry_id, model.model_id)]}
                    columns={perModelColumns}
                    includeCleanup={includeCleanup}
                  />
                ))}
              </tr>
              {expanded[entry.entry_id] && (
                <tr className="border-b border-mid-gray/10">
                  <td colSpan={totalColumns} className="px-2 py-2 space-y-2">
                    {models.map((model) => (
                      <DiffLine
                        key={model.model_id}
                        model={model}
                        result={
                          results[resultKey(entry.entry_id, model.model_id)]
                        }
                      />
                    ))}
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ResultCells: React.FC<{
  entry: ReplayBenchEntry;
  result: ReplayBenchResult | undefined;
  columns: number;
  includeCleanup: boolean;
}> = ({ entry, result, columns, includeCleanup }) => {
  const { t } = useTranslation();
  const first = `${tdClass} border-l border-mid-gray/20`;

  if (result?.failure) {
    return (
      <td
        colSpan={columns}
        className={`${first} text-red-500 whitespace-normal`}
      >
        {result.failure === "decode_failed" || !entry.readable
          ? t("settings.debug.replayBench.decodeFailed")
          : t("settings.debug.replayBench.transcribeFailed")}
      </td>
    );
  }

  return (
    <>
      <td className={first}>{formatPct(result?.differs_pct)}</td>
      <td className={tdClass}>{formatMs(result?.transcribe_ms)}</td>
      <td className={tdClass}>{formatRtf(result?.rtf)}</td>
      {includeCleanup && (
        <td className={tdClass}>
          {result?.cleanup_failed
            ? t("settings.debug.replayBench.cleanupFailed")
            : formatMs(result?.cleanup_ms)}
        </td>
      )}
    </>
  );
};

const DiffToken: React.FC<{ token: WordDiffToken }> = ({ token }) => {
  if (token.kind === "removed") {
    return <del className="text-red-500 line-through">{token.text}</del>;
  }
  if (token.kind === "added") {
    return (
      <ins className="text-green-600 underline decoration-2">{token.text}</ins>
    );
  }
  return <span>{token.text}</span>;
};

const DiffLine: React.FC<{
  model: ReplayBenchModel;
  result: ReplayBenchResult | undefined;
}> = ({ model, result }) => {
  const { t } = useTranslation();
  return (
    <div className="text-xs select-text">
      <span className="font-medium">{model.model_name}: </span>
      {!result ? (
        <span className="text-mid-gray">
          {t("settings.debug.replayBench.table.pending")}
        </span>
      ) : result.failure ? (
        <span className="text-red-500">
          {result.failure === "decode_failed"
            ? t("settings.debug.replayBench.decodeFailed")
            : t("settings.debug.replayBench.transcribeFailed")}
        </span>
      ) : result.diff.length === 0 ? (
        <span className="text-mid-gray">
          {t("settings.debug.replayBench.table.empty")}
        </span>
      ) : (
        result.diff.map((token, index) => (
          <React.Fragment key={index}>
            {index > 0 && " "}
            <DiffToken token={token} />
          </React.Fragment>
        ))
      )}
    </div>
  );
};
