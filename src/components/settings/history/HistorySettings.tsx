import React, { useCallback, useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { readFile } from "@tauri-apps/plugin-fs";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  FolderOpen,
  RotateCcw,
  Search,
  Star,
  Trash2,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  commands,
  events,
  type HistoryEntry,
  type HistoryUpdatePayload,
} from "@/bindings";
import { useOsType } from "@/hooks/useOsType";
import { formatDateTime } from "@/utils/dateFormat";
import { AudioPlayer, AudioPlayerGroup } from "../../ui/AudioPlayer";
import { Button } from "../../ui/Button";
import { copyToClipboard } from "./clipboard";
import {
  entryMatchesQuery,
  getEntryTexts,
  splitByQuery,
  type EntryStatus,
} from "./historyText";

const IconButton: React.FC<{
  onClick: () => void;
  title: string;
  disabled?: boolean;
  active?: boolean;
  children: React.ReactNode;
}> = ({ onClick, title, disabled, active, children }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`p-1.5 rounded-md flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed disabled:text-text/20 ${
      active
        ? "text-logo-primary hover:text-logo-primary/80"
        : "text-text/50 hover:text-logo-primary"
    }`}
    title={title}
  >
    {children}
  </button>
);

const PAGE_SIZE = 30;
const SEARCH_DEBOUNCE_MS = 150;

interface OpenRecordingsButtonProps {
  onClick: () => void;
  label: string;
}

const OpenRecordingsButton: React.FC<OpenRecordingsButtonProps> = ({
  onClick,
  label,
}) => (
  <Button
    onClick={onClick}
    variant="secondary"
    size="sm"
    className="flex items-center gap-2"
    title={label}
  >
    <FolderOpen className="w-4 h-4" />
    <span>{label}</span>
  </Button>
);

export const HistorySettings: React.FC = () => {
  const { t } = useTranslation();
  const osType = useOsType();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  // Bumped whenever a first page replaces the list, so the infinite-scroll
  // observer is recreated (and fires again if the sentinel is still visible).
  const [pageVersion, setPageVersion] = useState(0);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const activeQuery = debouncedQuery.trim();
  const sentinelRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const entriesRef = useRef<HistoryEntry[]>([]);
  const loadingRef = useRef(false);
  const activeQueryRef = useRef("");
  // Identifies the current list; responses for an older list are dropped.
  const generationRef = useRef(0);
  const initialLoadDoneRef = useRef(false);

  // Keep refs in sync for use in IntersectionObserver / event callbacks
  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  useEffect(() => {
    activeQueryRef.current = activeQuery;
  }, [activeQuery]);

  // Debounce typing before hitting the backend.
  useEffect(() => {
    const handle = setTimeout(
      () => setDebouncedQuery(query),
      SEARCH_DEBOUNCE_MS,
    );
    return () => clearTimeout(handle);
  }, [query]);

  const loadPage = useCallback(async (cursor?: number) => {
    const isFirstPage = cursor === undefined;
    if (!isFirstPage && loadingRef.current) return;
    loadingRef.current = true;

    if (isFirstPage) {
      generationRef.current += 1;
      setFetching(true);
      // Only the very first load swaps the list for a loading message; later
      // reloads (search edits) keep the current list until results arrive.
      if (!initialLoadDoneRef.current) setLoading(true);
    }
    const generation = generationRef.current;
    const searchQuery = activeQueryRef.current;

    try {
      const result = searchQuery
        ? await commands.searchHistoryEntries(
            searchQuery,
            cursor ?? null,
            PAGE_SIZE,
          )
        : await commands.getHistoryEntries(cursor ?? null, PAGE_SIZE);
      if (generation !== generationRef.current) return;
      if (result.status === "ok") {
        const { entries: newEntries, has_more } = result.data;
        setEntries((prev) =>
          isFirstPage ? newEntries : [...prev, ...newEntries],
        );
        setHasMore(has_more);
        if (isFirstPage) setPageVersion((v) => v + 1);
      } else {
        console.error("Failed to load history entries:", result.error);
      }
    } catch (error) {
      console.error("Failed to load history entries:", error);
    } finally {
      if (generation === generationRef.current) {
        setLoading(false);
        setFetching(false);
        initialLoadDoneRef.current = true;
      }
      loadingRef.current = false;
    }
  }, []);

  // Initial load, and a fresh first page whenever the search changes.
  useEffect(() => {
    activeQueryRef.current = activeQuery;
    loadPage();
  }, [activeQuery, loadPage]);

  // Infinite scroll via IntersectionObserver
  useEffect(() => {
    if (loading) return;

    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore) return;

    const observer = new IntersectionObserver(
      (observerEntries) => {
        const first = observerEntries[0];
        if (first.isIntersecting) {
          const lastEntry = entriesRef.current[entriesRef.current.length - 1];
          if (lastEntry) {
            loadPage(lastEntry.id);
          }
        }
      },
      { threshold: 0 },
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loading, hasMore, loadPage, pageVersion]);

  // Listen for new entries added from the transcription pipeline
  useEffect(() => {
    const unlisten = events.historyUpdatePayload.listen((event) => {
      const payload: HistoryUpdatePayload = event.payload;
      if (payload.action === "added") {
        // While searching, only show new dictations that match.
        if (entryMatchesQuery(payload.entry, activeQueryRef.current)) {
          setEntries((prev) => [payload.entry, ...prev]);
        }
      } else if (payload.action === "updated") {
        setEntries((prev) =>
          prev.map((e) => (e.id === payload.entry.id ? payload.entry : e)),
        );
      }
      // "deleted" and "toggled" are handled by optimistic updates only,
      // so we intentionally ignore them here to avoid double-mutation.
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  // Ctrl+F (Cmd+F on macOS) focuses the search box while this page is shown.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        (event.ctrlKey || event.metaKey) &&
        !event.shiftKey &&
        !event.altKey &&
        event.key.toLowerCase() === "f"
      ) {
        event.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  const clearSearch = () => {
    setQuery("");
    setDebouncedQuery("");
  };

  const handleSearchKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>,
  ) => {
    if (event.key === "Escape" && query !== "") {
      event.preventDefault();
      event.stopPropagation();
      clearSearch();
    }
  };

  const toggleSaved = async (id: number) => {
    // Optimistic update
    setEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
    );
    try {
      const result = await commands.toggleHistoryEntrySaved(id);
      if (result.status !== "ok") {
        // Revert on failure
        setEntries((prev) =>
          prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
        );
      }
    } catch (error) {
      console.error("Failed to toggle saved status:", error);
      // Revert on failure
      setEntries((prev) =>
        prev.map((e) => (e.id === id ? { ...e, saved: !e.saved } : e)),
      );
    }
  };

  const getAudioUrl = useCallback(
    async (fileName: string) => {
      try {
        const result = await commands.getAudioFilePath(fileName);
        if (result.status === "ok") {
          if (osType === "linux") {
            const fileData = await readFile(result.data);
            const blob = new Blob([fileData], { type: "audio/wav" });
            return URL.createObjectURL(blob);
          }
          return convertFileSrc(result.data, "asset");
        }
        return null;
      } catch (error) {
        console.error("Failed to get audio file path:", error);
        return null;
      }
    },
    [osType],
  );

  const deleteAudioEntry = async (id: number) => {
    // Optimistically remove
    setEntries((prev) => prev.filter((e) => e.id !== id));
    try {
      const result = await commands.deleteHistoryEntry(id);
      if (result.status !== "ok") {
        // Reload on failure
        loadPage();
      }
    } catch (error) {
      console.error("Failed to delete entry:", error);
      loadPage();
    }
  };

  const retryHistoryEntry = async (id: number) => {
    const result = await commands.retryHistoryEntryTranscription(id);
    if (result.status !== "ok") {
      throw new Error(String(result.error));
    }
  };

  const openRecordingsFolder = async () => {
    try {
      const result = await commands.openRecordingsFolder();
      if (result.status !== "ok") {
        throw new Error(String(result.error));
      }
    } catch (error) {
      console.error("Failed to open recordings folder:", error);
    }
  };

  let content: React.ReactNode;

  if (loading) {
    content = (
      <div className="px-4 py-3 text-center text-text/60">
        {t("settings.history.loading")}
      </div>
    );
  } else if (entries.length === 0 && activeQuery) {
    content = fetching ? (
      <div className="px-4 py-3 text-center text-text/60">
        {t("settings.history.searching")}
      </div>
    ) : (
      <div className="px-4 py-3 flex flex-col items-center gap-2 text-center text-text/60">
        <p>{t("settings.history.noMatches", { query: activeQuery })}</p>
        <Button variant="secondary" size="sm" onClick={clearSearch}>
          {t("settings.history.clearSearch")}
        </Button>
      </div>
    );
  } else if (entries.length === 0) {
    content = (
      <div className="px-4 py-3 text-center text-text/60">
        {t("settings.history.empty")}
      </div>
    );
  } else {
    content = (
      <>
        <AudioPlayerGroup>
          <div className="divide-y divide-mid-gray/20">
            {entries.map((entry) => (
              <HistoryEntryComponent
                key={entry.id}
                entry={entry}
                query={activeQuery}
                onToggleSaved={() => toggleSaved(entry.id)}
                getAudioUrl={getAudioUrl}
                deleteAudio={deleteAudioEntry}
                retryTranscription={retryHistoryEntry}
              />
            ))}
          </div>
        </AudioPlayerGroup>
        {/* Sentinel for infinite scroll */}
        <div ref={sentinelRef} className="h-1" />
      </>
    );
  }

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="space-y-2">
        <div className="sticky top-0 z-10 bg-background pb-2">
          <div className="relative">
            <Search
              width={16}
              height={16}
              aria-hidden="true"
              className="absolute start-3 top-1/2 -translate-y-1/2 text-text/50 pointer-events-none"
            />
            <input
              ref={searchInputRef}
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder={t("settings.history.searchPlaceholder")}
              aria-label={t("settings.history.searchPlaceholder")}
              className="w-full ps-9 pe-3 py-2 text-sm bg-mid-gray/10 border border-mid-gray/40 rounded-md transition-colors hover:border-logo-primary focus:outline-none focus:border-logo-primary focus:bg-logo-primary/10"
            />
          </div>
        </div>
        <div className="px-4 flex items-center justify-between">
          <div>
            <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
              {t("settings.history.title")}
            </h2>
          </div>
          <OpenRecordingsButton
            onClick={openRecordingsFolder}
            label={t("settings.history.openFolder")}
          />
        </div>
        <div className="bg-background border border-mid-gray/20 rounded-lg overflow-visible">
          {content}
        </div>
      </div>
    </div>
  );
};

/** Renders `text` with case-insensitive matches of `query` wrapped in <mark>. */
const Highlighted: React.FC<{ text: string; query: string }> = ({
  text,
  query,
}) => (
  <>
    {splitByQuery(text, query).map((part, index) =>
      part.match ? (
        <mark
          key={index}
          className="bg-logo-primary/30 text-inherit rounded-sm px-0.5"
        >
          {part.text}
        </mark>
      ) : (
        <React.Fragment key={index}>{part.text}</React.Fragment>
      ),
    )}
  </>
);

const STATUS_LABEL_KEYS: Record<EntryStatus, string> = {
  cleanedUp: "settings.history.status.cleanedUp",
  cleanedUpLate: "settings.history.status.cleanedUpLate",
  cleaningUp: "settings.history.status.cleaningUp",
  original: "settings.history.status.original",
  cleanupFailed: "settings.history.status.cleanupFailed",
};

const StatusBadge: React.FC<{ status: EntryStatus }> = ({ status }) => {
  const { t } = useTranslation();
  const tone =
    status === "cleanupFailed"
      ? "border-red-500/40 text-red-600 dark:text-red-400"
      : status === "cleanedUp" || status === "cleanedUpLate"
        ? "border-logo-primary/40 text-text/80"
        : "border-mid-gray/40 text-text/60";
  return (
    <span
      className={`text-[11px] leading-none font-medium px-1.5 py-1 rounded border ${tone}`}
    >
      {t(STATUS_LABEL_KEYS[status])}
    </span>
  );
};

interface HistoryEntryProps {
  entry: HistoryEntry;
  query: string;
  onToggleSaved: () => void;
  getAudioUrl: (fileName: string) => Promise<string | null>;
  deleteAudio: (id: number) => Promise<void>;
  retryTranscription: (id: number) => Promise<void>;
}

const HistoryEntryComponent: React.FC<HistoryEntryProps> = ({
  entry,
  query,
  onToggleSaved,
  getAudioUrl,
  deleteAudio,
  retryTranscription,
}) => {
  const { t, i18n } = useTranslation();
  const [showCopied, setShowCopied] = useState(false);
  const [showCopiedOriginal, setShowCopiedOriginal] = useState(false);
  const [retrying, setRetrying] = useState(false);
  // null = follow the default (expanded only when the search matched the
  // original text alone); a click pins the user's choice.
  const [expandedChoice, setExpandedChoice] = useState<boolean | null>(null);

  const { primaryText, originalText, hasDistinctOriginal, status } =
    getEntryTexts(entry);
  const hasTranscription = primaryText.trim().length > 0;
  const originalOnlyMatch =
    hasDistinctOriginal &&
    query.length > 0 &&
    splitByQuery(primaryText, query).every((part) => !part.match) &&
    splitByQuery(originalText, query).some((part) => part.match);
  const expanded = hasDistinctOriginal && (expandedChoice ?? originalOnlyMatch);
  const originalRegionId = `history-original-${entry.id}`;

  const handleLoadAudio = useCallback(
    () => getAudioUrl(entry.file_name),
    [getAudioUrl, entry.file_name],
  );

  const copyText = async (text: string, onCopied: (v: boolean) => void) => {
    if (text.trim().length === 0) {
      return;
    }

    const copied = await copyToClipboard(text);
    if (!copied) {
      toast.error(t("settings.history.copyError"));
      return;
    }

    onCopied(true);
    setTimeout(() => onCopied(false), 2000);
  };

  const handleDeleteEntry = async () => {
    try {
      await deleteAudio(entry.id);
    } catch (error) {
      console.error("Failed to delete entry:", error);
      toast.error(t("settings.history.deleteError"));
    }
  };

  const handleRetranscribe = async () => {
    try {
      setRetrying(true);
      await retryTranscription(entry.id);
    } catch (error) {
      console.error("Failed to re-transcribe:", error);
      toast.error(t("settings.history.retranscribeError"));
    } finally {
      setRetrying(false);
    }
  };

  const formattedDate = formatDateTime(String(entry.timestamp), i18n.language);

  return (
    <div className="px-4 py-2 pb-5 flex flex-col gap-3">
      <div className="flex justify-between items-center gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <p className="text-sm font-medium">{formattedDate}</p>
          {status && !retrying && <StatusBadge status={status} />}
        </div>
        <div className="flex items-center">
          <IconButton
            onClick={() => copyText(primaryText, setShowCopied)}
            disabled={!hasTranscription || retrying}
            title={
              hasDistinctOriginal
                ? t("settings.history.copyPolished")
                : t("settings.history.copy")
            }
          >
            {showCopied ? (
              <Check width={16} height={16} />
            ) : (
              <Copy width={16} height={16} />
            )}
          </IconButton>
          <IconButton
            onClick={onToggleSaved}
            disabled={retrying}
            active={entry.saved}
            title={
              entry.saved
                ? t("settings.history.unsave")
                : t("settings.history.save")
            }
          >
            <Star
              width={16}
              height={16}
              fill={entry.saved ? "currentColor" : "none"}
            />
          </IconButton>
          <IconButton
            onClick={handleRetranscribe}
            disabled={retrying}
            title={t("settings.history.retranscribe")}
          >
            <RotateCcw
              width={16}
              height={16}
              style={
                retrying
                  ? { animation: "spin 1s linear infinite reverse" }
                  : undefined
              }
            />
          </IconButton>
          <IconButton
            onClick={handleDeleteEntry}
            disabled={retrying}
            title={t("settings.history.delete")}
          >
            <Trash2 width={16} height={16} />
          </IconButton>
        </div>
      </div>

      <p
        className={`italic text-sm ${
          retrying
            ? ""
            : hasTranscription
              ? "text-text/90 select-text cursor-text whitespace-pre-wrap break-words"
              : "text-text/40"
        }`}
        style={
          retrying
            ? { animation: "transcribe-pulse 3s ease-in-out infinite" }
            : undefined
        }
      >
        {retrying && (
          <style>{`
            @keyframes transcribe-pulse {
              0%, 100% { color: color-mix(in srgb, var(--color-text) 40%, transparent); }
              50% { color: color-mix(in srgb, var(--color-text) 90%, transparent); }
            }
          `}</style>
        )}
        {retrying ? (
          t("settings.history.transcribing")
        ) : hasTranscription ? (
          <Highlighted text={primaryText} query={query} />
        ) : (
          t("settings.history.transcriptionFailed")
        )}
      </p>

      {hasDistinctOriginal && !retrying && (
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={() => setExpandedChoice(!expanded)}
            aria-expanded={expanded}
            aria-controls={originalRegionId}
            className="self-start flex items-center gap-1 text-xs text-text/60 hover:text-logo-primary cursor-pointer"
          >
            {expanded ? (
              <ChevronDown width={14} height={14} aria-hidden="true" />
            ) : (
              <ChevronRight width={14} height={14} aria-hidden="true" />
            )}
            {expanded
              ? t("settings.history.hideOriginal")
              : t("settings.history.showOriginal")}
          </button>
          {expanded && (
            <div
              id={originalRegionId}
              className="flex flex-col gap-2 border-s-2 border-mid-gray/30 ps-3"
            >
              <span className="text-xs font-medium text-text/60">
                {t("settings.history.originalText")}
              </span>
              <p className="text-sm text-text/70 select-text cursor-text whitespace-pre-wrap break-words">
                <Highlighted text={originalText} query={query} />
              </p>
              <Button
                variant="secondary"
                size="sm"
                className="self-start flex items-center gap-2"
                onClick={() => copyText(originalText, setShowCopiedOriginal)}
              >
                {showCopiedOriginal ? (
                  <Check width={14} height={14} aria-hidden="true" />
                ) : (
                  <Copy width={14} height={14} aria-hidden="true" />
                )}
                <span>{t("settings.history.copyOriginal")}</span>
              </Button>
            </div>
          )}
        </div>
      )}

      <AudioPlayer onLoadRequest={handleLoadAudio} className="w-full" />
    </div>
  );
};
