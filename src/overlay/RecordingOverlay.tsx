import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands, events } from "@/bindings";
import type {
  StreamPhase,
  StreamPhaseEvent,
  StreamTextEvent,
  StreamWorkKind,
} from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";
import { type OverlayNotice, truncateParams } from "./notice";

type OverlayState =
  | "recording"
  | "streaming"
  | "transcribing"
  | "processing"
  | "notice";

/** What the running cleanup shares (overlay.rs `emit_cleanup_context`). */
interface CleanupContext {
  app: string | null;
  screenshot: boolean;
}

// Number of reactive bars in the waveform (the simple, smoothed style shared by
// every overlay form). Mic levels arrive as 16 FFT buckets; we take the first N.
const WAVE_BARS = 9;

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  // `Stream::play()` returning does not mean hardware callbacks are flowing.
  // Stay visually in an arming state until the backend processes the first
  // actual microphone sample chunk.
  const [captureReady, setCaptureReady] = useState(false);
  const [levels, setLevels] = useState<number[]>(Array(WAVE_BARS).fill(0));
  const [streamText, setStreamText] = useState<StreamTextEvent>({
    committed: "",
    tentative: "",
  });
  const [phase, setPhase] = useState<StreamPhase>("listening");
  const [workKind, setWorkKind] = useState<StreamWorkKind>("transcribing");
  const [elapsed, setElapsed] = useState(0);
  // Bumped on each new streaming session so the Live card remounts fresh (replays
  // the pop-in, and never animates in from the previous panel's open size).
  const [session, setSession] = useState(0);
  // Overlay placement (top vs bottom of the screen). The Live panel grows downward
  // from a top overlay (oldest line under the pill) and upward from a bottom one.
  const [position, setPosition] = useState<"top" | "bottom">("bottom");
  // True once live text overflows the cap. A top overlay fades its top edge only
  // while overflowing, so the resting first line stays crisp flush under the pill.
  const [overflowing, setOverflowing] = useState(false);

  const smoothedLevelsRef = useRef<number[]>(Array(16).fill(0));
  // Live-text scroll-back: the text region "sticks" to the newest line while the
  // user is at the bottom; if they scroll up to read history, auto-follow pauses
  // until they scroll back down.
  const capRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const direction = getLanguageDirection(i18n.language);

  // Transient notice (see overlay_notice.rs). The backend hides the overlay
  // when we report the timer ran out; hovering pauses the timer.
  const [notice, setNotice] = useState<OverlayNotice | null>(null);
  const [noticeHovered, setNoticeHovered] = useState(false);
  const [noticeActionPending, setNoticeActionPending] = useState(false);
  // Time left on the current notice, keyed by id so a pausing cleanup can't
  // eat into the next notice's budget.
  const noticeRemainingRef = useRef({ id: 0, ms: 0 });
  // Show events finish asynchronously (settings I/O); only the newest one may
  // apply its state, so a slow notice can never override a newer press.
  const showSeqRef = useRef(0);
  // App context chip for "Cleaning up... · Slack" (+ camera if a screenshot
  // was shared). Cleared whenever a new dictation starts.
  const [cleanupContext, setCleanupContext] = useState<CleanupContext | null>(
    null,
  );

  useEffect(() => {
    const readPlacement = async () => {
      // The Live panel flows downward from a top overlay and upward from a
      // bottom one; read the placement so the layout can flip to match.
      try {
        const settings = await commands.getAppSettings();
        if (settings.status === "ok") {
          setPosition(
            settings.data.overlay_position === "top" ? "top" : "bottom",
          );
        }
      } catch {
        // Keep the previous/default placement if settings can't be read.
      }
    };

    const setupEventListeners = async () => {
      const unlistenShow = await listen("show-overlay", async (event) => {
        const overlayState = event.payload as OverlayState;
        const seq = ++showSeqRef.current;
        // A dictation state always replaces a notice.
        setNotice(null);
        // Reset synchronously before settings I/O. A fast microphone can emit
        // recording-ready while the awaits below are in flight; resetting after
        // them would overwrite that event and leave the overlay stuck arming.
        if (
          overlayState === "recording" ||
          overlayState === "streaming" ||
          overlayState === "transcribing"
        ) {
          setCleanupContext(null);
        }
        if (overlayState === "recording" || overlayState === "streaming") {
          setCaptureReady(false);
          smoothedLevelsRef.current = Array(16).fill(0);
          setLevels(Array(WAVE_BARS).fill(0));
          setStreamText({ committed: "", tentative: "" });
        }

        await syncLanguageFromSettings();
        await readPlacement();
        if (seq !== showSeqRef.current) return;
        setState(overlayState);
        if (overlayState === "streaming") {
          setPhase("listening");
          setWorkKind("transcribing");
          setElapsed(0);
          setSession((s) => s + 1); // remount the card fresh for this session
        }
        setIsVisible(true);
      });

      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
        setCaptureReady(false);
      });

      const unlistenNotice = await listen<OverlayNotice>(
        "overlay-notice",
        async (event) => {
          const seq = ++showSeqRef.current;
          await syncLanguageFromSettings();
          await readPlacement();
          if (seq !== showSeqRef.current) return;
          noticeRemainingRef.current = {
            id: event.payload.id,
            ms: event.payload.duration_ms,
          };
          setNoticeHovered(false);
          setNoticeActionPending(false);
          setNotice(event.payload);
          setState("notice");
          setIsVisible(true);
        },
      );

      const unlistenContext = await listen<CleanupContext>(
        "cleanup-context",
        (event) => {
          setCleanupContext(event.payload);
        },
      );

      const unlistenReady = await listen("recording-ready", () => {
        setElapsed(0);
        setCaptureReady(true);
      });

      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const newLevels = event.payload as number[];
        // Exponential smoothing across the 16 buckets, then take the first N
        // bars for the shared waveform.
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = newLevels[i] || 0;
          return prev * 0.7 + target * 0.3;
        });
        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed.slice(0, WAVE_BARS));
      });

      const unlistenStream = await events.streamTextEvent.listen((event) => {
        setStreamText(event.payload);
      });

      const unlistenPhase = await events.streamPhaseEvent.listen((event) => {
        const payload: StreamPhaseEvent = event.payload;
        setPhase(payload.phase);
        if (payload.kind) setWorkKind(payload.kind);
      });

      return () => {
        unlistenShow();
        unlistenHide();
        unlistenNotice();
        unlistenContext();
        unlistenReady();
        unlistenLevel();
        unlistenStream();
        unlistenPhase();
      };
    };

    setupEventListeners();
  }, []);

  // Elapsed capture timer starts only once microphone samples are flowing.
  useEffect(() => {
    if (state !== "streaming" || !isVisible || !captureReady) return;
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [state, isVisible, captureReady]);

  // Notice timer: counts down only while the notice is shown, not hovered and
  // no action is running; pausing keeps the remaining time.
  useEffect(() => {
    if (state !== "notice" || !notice || !isVisible) return;
    if (noticeHovered || noticeActionPending) return;
    const started = Date.now();
    const remaining = noticeRemainingRef.current;
    const timer = setTimeout(
      () => {
        commands.dismissOverlayNotice(notice.id).catch((e) => {
          console.warn("Failed to dismiss overlay notice:", e);
        });
      },
      remaining.id === notice.id ? remaining.ms : notice.duration_ms,
    );
    return () => {
      clearTimeout(timer);
      const current = noticeRemainingRef.current;
      if (current.id === notice.id) {
        current.ms = Math.max(0, current.ms - (Date.now() - started));
      }
    };
  }, [state, notice, isVisible, noticeHovered, noticeActionPending]);

  // Stick to the bottom as text streams in — but only while pinned, so a user who
  // has scrolled up to read history isn't yanked back down by the next chunk.
  useLayoutEffect(() => {
    const el = capRef.current;
    if (!el) return;
    // Fade the top edge only once text actually overflows the cap.
    setOverflowing(el.scrollHeight > el.clientHeight + 1);
    if (pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [streamText]);

  // Each fresh streaming session starts pinned to the bottom, fade cleared.
  useEffect(() => {
    pinnedRef.current = true;
    setOverflowing(false);
  }, [session]);

  if (!isVisible) return null;

  // Re-pin when the user is within ~a line of the bottom; unpin otherwise.
  const handleStreamScroll = () => {
    const el = capRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 16;
  };

  const fmtTime = (s: number) =>
    `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  // ---- Shared building blocks (one visual language for every overlay form) ----
  const waveform = (
    <div className={`swave ${captureReady ? "ready" : "arming"}`}>
      {levels.map((v, i) => (
        <i
          key={i}
          style={{
            height: `${Math.max(3, Math.min(18, 3 + Math.pow(v, 0.7) * 15))}px`,
          }}
        />
      ))}
    </div>
  );

  const cancelBtn = (
    <button
      className="sx"
      aria-label="cancel"
      onClick={() => commands.cancelOperation()}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path
          d="M4 4 L12 12 M12 4 L4 12"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    </button>
  );

  // dot (left) | waveform (center) | timer + cancel (right) — same structure for
  // pill & panel, so the Live morph is a pure width change.
  const listeningRow = (showTimer: boolean, showCancel: boolean) => (
    <div className="sbase">
      <div className="sbase-l">
        <span className={`sdot ${captureReady ? "ready" : "arming"}`} />
      </div>
      {waveform}
      <div className="sbase-r">
        {showTimer && <span className="stimer">{fmtTime(elapsed)}</span>}
        {showCancel && cancelBtn}
      </div>
    </div>
  );

  // spinner (left) | label (center) | cancel (right) — same 3-zone grid as the
  // listening row, so the label is centered.
  const workingRow = (
    label: string,
    showCancel: boolean,
    screenshotShared = false,
  ) => (
    <div className="sbase">
      <div className="sbase-l">
        <span className="sspinner" />
      </div>
      <span className={`swork-label ${screenshotShared ? "with-icon" : ""}`}>
        <span className="swork-text">{label}</span>
        {screenshotShared && (
          <span
            className="scamera"
            role="img"
            aria-label={t("overlay.screenshotShared")}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M2 5.2 H4.6 L5.8 3.6 H10.2 L11.4 5.2 H14 V12.6 H2 Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinejoin="round"
              />
              <circle
                cx="8"
                cy="8.8"
                r="2.2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
              />
            </svg>
          </span>
        )}
      </span>
      <div className="sbase-r">{showCancel && cancelBtn}</div>
    </div>
  );

  // "Cleaning up..." plus the app chip when app info is shared.
  const cleanupLabel = cleanupContext?.app
    ? t("overlay.processingWithApp", { app: cleanupContext.app })
    : t("overlay.processing");
  const cleanupScreenshot = cleanupContext?.screenshot ?? false;
  const hasCleanupChip = Boolean(cleanupContext?.app) || cleanupScreenshot;

  // ---- Notice: icon + one short line + optional action button ----
  if (state === "notice" && notice) {
    const isWarning = notice.kind === "warning";
    const message = t(
      notice.message.key,
      truncateParams(notice.message.params),
    );
    const fullMessage = t(notice.message.key, notice.message.params);
    const action = notice.action;
    const runAction = () => {
      if (noticeActionPending) return;
      setNoticeActionPending(true);
      commands.runOverlayNoticeAction(notice.id).catch((e) => {
        console.warn("Failed to run overlay notice action:", e);
      });
    };

    return (
      <div
        dir={direction}
        className={`ov-stage ${position} ov-fade ${isVisible ? "show" : ""}`}
      >
        <div
          className={`scard notice ${notice.kind}`}
          onMouseEnter={() => setNoticeHovered(true)}
          onMouseLeave={() => setNoticeHovered(false)}
        >
          <span
            className="nicon"
            role="img"
            aria-label={
              isWarning ? t("overlay.notice.warning") : t("overlay.notice.info")
            }
          >
            {isWarning ? (
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path
                  d="M8 2.2 L14.4 13.4 H1.6 Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
                <path
                  d="M8 6.4 V9.4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
                <circle cx="8" cy="11.4" r="0.85" fill="currentColor" />
              </svg>
            ) : (
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <circle
                  cx="8"
                  cy="8"
                  r="6.2"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M8 7.2 V11.2"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
                <circle cx="8" cy="4.9" r="0.85" fill="currentColor" />
              </svg>
            )}
          </span>
          <span
            className="nmsg"
            role={notice.urgent ? "alert" : "status"}
            aria-live={notice.urgent ? "assertive" : "polite"}
            title={fullMessage !== message ? fullMessage : undefined}
          >
            {message}
          </span>
          {action && (
            <button
              className="naction"
              onClick={runAction}
              disabled={noticeActionPending}
              title={t(action.key, action.params)}
            >
              {t(action.key, truncateParams(action.params))}
            </button>
          )}
        </div>
      </div>
    );
  }

  // ---- Live overlay: a pill that sculpts open into a panel ----
  if (state === "streaming") {
    const hasText =
      streamText.committed.length > 0 || streamText.tentative.length > 0;
    const working = phase === "working";
    // Keep the panel open whenever there's text — even while finalizing — so the
    // transcript stays put under a working spinner instead of collapsing and
    // squishing the text mid-stream. Only fall back to the small working pill
    // when there was no text to preserve.
    const open = hasText;
    const collapsed = working && !hasText;
    const polishing = working && workKind === "polishing";

    return (
      <div dir={direction} className={`ov-stage ${position}`}>
        <div
          key={session}
          className={`scard ${open ? "open" : ""} ${collapsed ? "working" : ""} ${
            polishing && hasCleanupChip ? "ctx" : ""
          } ${isVisible ? "" : "leaving"}`}
        >
          <div className="stext">
            <div className="stext-clip">
              <div
                className={`stext-cap ${overflowing ? "overflowing" : ""}`}
                ref={capRef}
                onScroll={handleStreamScroll}
              >
                <p>
                  <span className="committed">
                    {streamText.committed ? streamText.committed + " " : ""}
                  </span>
                  <span className="tentative">{streamText.tentative}</span>
                  {/* Drop the blinking caret once finalizing — it's no longer
                      capturing, and a static spinner conveys the work. */}
                  {!working && <span className="scaret" />}
                </p>
              </div>
            </div>
          </div>
          {working
            ? polishing
              ? workingRow(cleanupLabel, true, cleanupScreenshot)
              : workingRow(t("overlay.transcribing"), true)
            : listeningRow(open, true)}
        </div>
      </div>
    );
  }

  // ---- Minimal overlay: exactly one row at a time — waveform (recording), or a
  // spinner + label (transcribing / processing). Never both. The pill animates its
  // width between them; the cancel button is in both rows so it stays put.
  const working = state === "transcribing" || state === "processing";
  const processing = state === "processing";
  const workLabel = processing ? cleanupLabel : t("overlay.transcribing");

  return (
    <div
      dir={direction}
      className={`ov-stage ${position} ov-fade ${isVisible ? "show" : ""}`}
    >
      <div
        className={`scard compact ${working && isVisible ? "cworking" : ""} ${
          processing && hasCleanupChip ? "ctx" : ""
        }`}
      >
        {working
          ? workingRow(workLabel, true, processing && cleanupScreenshot)
          : listeningRow(false, true)}
      </div>
    </div>
  );
};

export default RecordingOverlay;
