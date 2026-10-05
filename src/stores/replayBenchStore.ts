import { create } from "zustand";
import {
  commands,
  events,
  type ReplayBenchEntry,
  type ReplayBenchEvent,
  type ReplayBenchModel,
  type ReplayBenchModelSummary,
  type ReplayBenchPhase,
  type ReplayBenchRequest,
} from "@/bindings";
import {
  resultKey,
  type BenchResults,
} from "@/components/settings/debug/replay-bench/benchUtils";

export type BenchStatus = "idle" | "starting" | "running" | "done" | "stopped";

export interface BenchProgress {
  modelName: string;
  entryIndex: number;
  entryTotal: number;
  phase: ReplayBenchPhase;
}

interface ReplayBenchStore {
  status: BenchStatus;
  entries: ReplayBenchEntry[];
  models: ReplayBenchModel[];
  includeCleanup: boolean;
  results: BenchResults;
  summaries: Record<string, ReplayBenchModelSummary>;
  progress: BenchProgress | null;
  error: string | null;
  initialized: boolean;

  initialize: () => Promise<void>;
  start: (request: ReplayBenchRequest) => Promise<void>;
  stop: () => Promise<void>;
}

/**
 * Bench state lives in a store (not the component) so a run keeps reporting
 * while the user switches settings pages. Results are never persisted.
 */
export const useReplayBenchStore = create<ReplayBenchStore>()((set, get) => {
  let initializing: Promise<void> | null = null;
  // Bumped on every bench event, so a state query can tell whether an event
  // (e.g. "finished") overtook it and its answer is already stale.
  let eventSeq = 0;

  const handleEvent = (event: ReplayBenchEvent) => {
    eventSeq += 1;
    switch (event.type) {
      case "started":
        set({
          status: "running",
          entries: event.entries,
          models: event.models,
          includeCleanup: event.include_cleanup,
          results: {},
          summaries: {},
          progress: null,
          error: null,
        });
        break;
      case "progress":
        set({
          status: "running",
          progress: {
            modelName: event.model_name,
            entryIndex: event.entry_index,
            entryTotal: event.entry_total,
            phase: event.phase,
          },
        });
        break;
      case "result":
        set((state) => ({
          results: {
            ...state.results,
            [resultKey(event.result.entry_id, event.result.model_id)]:
              event.result,
          },
        }));
        break;
      case "model_finished":
        set((state) => ({
          summaries: {
            ...state.summaries,
            [event.summary.model_id]: event.summary,
          },
        }));
        break;
      case "finished":
        set({
          status: event.cancelled ? "stopped" : "done",
          progress: null,
        });
        break;
    }
  };

  return {
    status: "idle",
    entries: [],
    models: [],
    includeCleanup: false,
    results: {},
    summaries: {},
    progress: null,
    error: null,
    initialized: false,

    initialize: async () => {
      if (get().initialized) return;
      // Concurrent callers share one attempt; a failed attempt can be retried.
      if (initializing === null) {
        initializing = (async () => {
          try {
            await events.replayBenchEvent.listen((event) =>
              handleEvent(event.payload),
            );
            set({ initialized: true });
          } catch (e) {
            console.error("Failed to listen for replay bench events:", e);
            return;
          } finally {
            initializing = null;
          }
          // Reconcile with the backend: a run may have started (or ended)
          // before the listener existed.
          try {
            const seqBefore = eventSeq;
            const running = await commands.isReplayBenchRunning();
            // Events that arrived while the query was in flight are newer
            // than its answer; trust them instead.
            if (eventSeq !== seqBefore) return;
            if (running) {
              set({ status: "running" });
            } else if (get().status === "running") {
              set({ status: "idle", progress: null });
            }
          } catch (e) {
            console.error("Failed to query replay bench state:", e);
          }
        })();
      }
      await initializing;
    },

    start: async (request) => {
      // Listen first so the "started" event can't be missed.
      await get().initialize();
      set({ status: "starting", error: null });
      try {
        const result = await commands.startReplayBench(request);
        if (result.status === "error") {
          set({ status: "idle", error: result.error });
        }
      } catch (e) {
        set({ status: "idle", error: String(e) });
      }
    },

    stop: async () => {
      try {
        await commands.stopReplayBench();
        // If the backend run already ended without a "finished" event
        // reaching us, don't leave the page stuck on "running".
        const seqBefore = eventSeq;
        const running = await commands.isReplayBenchRunning();
        if (!running && eventSeq === seqBefore) {
          const { status } = get();
          if (status === "running" || status === "starting") {
            set({ status: "stopped", progress: null });
          }
        }
      } catch (e) {
        console.error("Failed to stop replay bench:", e);
      }
    },
  };
});
