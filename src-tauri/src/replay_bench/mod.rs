//! Personal replay bench (Debug page): re-run saved history recordings
//! through installed models and report how much each model's text disagrees
//! with what Handy saved, plus load/transcribe/cleanup latency.
//!
//! The bench never touches the live pipeline. Each model is loaded into its
//! own [`IsolatedEngine`] (the active model stays loaded for dictation), runs
//! on a background thread, and is dropped before the next model loads so at
//! most one bench model is resident. A dictation press pauses the bench
//! between clips until the pipeline is idle again.

pub mod metrics;

use crate::cockpit::deadline::{CleanupOutcome, RequestSent};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::{HistoryEntry, HistoryManager};
use crate::managers::model::ModelManager;
use crate::managers::transcription::IsolatedEngine;
use crate::settings::{get_settings, AppSettings};
use crate::TranscriptionCoordinator;
use log::{error, info, warn};
use metrics::{
    mean, median, percentile, real_time_factor, reference_text, select_entries, word_diff,
    Candidate, ReplayBenchSelection, WordDiffToken,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_specta::Event;

const PAUSE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const SAMPLE_RATE: f64 = 16_000.0;

/// Only one bench runs at a time.
static RUNNING: AtomicBool = AtomicBool::new(false);
/// Set by Stop; checked between clips, between models and while paused.
static CANCEL: AtomicBool = AtomicBool::new(false);

// What to replay and against which models.
#[derive(Clone, Debug, Deserialize, Type)]
pub struct ReplayBenchRequest {
    pub selection: ReplayBenchSelection,
    pub model_ids: Vec<String>,
    pub include_cleanup: bool,
}

// One recording in the run. `reference_text` is what the results are
// compared against.
#[derive(Clone, Debug, Serialize, Type)]
pub struct ReplayBenchEntry {
    pub entry_id: i64,
    pub timestamp: i64,
    pub reference_text: String,
    pub duration_secs: f64,
    pub readable: bool,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ReplayBenchModel {
    pub model_id: String,
    pub model_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBenchPhase {
    Loading,
    Transcribing,
    Cleanup,
    Paused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBenchFailure {
    DecodeFailed,
    TranscribeFailed,
}

// One recording replayed through one model.
#[derive(Clone, Debug, Serialize, Type)]
pub struct ReplayBenchResult {
    pub entry_id: i64,
    pub model_id: String,
    pub failure: Option<ReplayBenchFailure>,
    pub text: String,
    pub cleanup_text: Option<String>,
    pub cleanup_failed: bool,
    pub differs_pct: Option<f64>,
    pub diff: Vec<WordDiffToken>,
    pub transcribe_ms: Option<f64>,
    pub rtf: Option<f64>,
    pub cleanup_ms: Option<f64>,
}

// Per-model roll-up, emitted once the model is done (or failed to load).
#[derive(Clone, Debug, Serialize, Type)]
pub struct ReplayBenchModelSummary {
    pub model_id: String,
    pub load_ms: Option<f64>,
    pub load_error: Option<String>,
    pub transcribe_median_ms: Option<f64>,
    pub transcribe_p90_ms: Option<f64>,
    pub rtf_median: Option<f64>,
    pub cleanup_median_ms: Option<f64>,
    pub cleanup_p90_ms: Option<f64>,
    pub mean_differs_pct: Option<f64>,
    pub completed: u32,
    pub failed: u32,
}

#[derive(Clone, Debug, Serialize, Type, tauri_specta::Event)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplayBenchEvent {
    Started {
        entries: Vec<ReplayBenchEntry>,
        models: Vec<ReplayBenchModel>,
        include_cleanup: bool,
    },
    Progress {
        model_id: String,
        model_name: String,
        entry_index: u32,
        entry_total: u32,
        phase: ReplayBenchPhase,
    },
    Result {
        result: ReplayBenchResult,
    },
    ModelFinished {
        summary: ReplayBenchModelSummary,
    },
    Finished {
        cancelled: bool,
    },
}

/// Clears [`RUNNING`] however the worker exits, including a panic.
struct RunningGuard;

impl Drop for RunningGuard {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::Release);
    }
}

pub fn is_running() -> bool {
    RUNNING.load(Ordering::Acquire)
}

pub fn request_stop() {
    if is_running() {
        CANCEL.store(true, Ordering::Release);
    }
}

fn cancelled() -> bool {
    CANCEL.load(Ordering::Acquire)
}

/// Pick the recordings a selection would replay. Only the entries the
/// selection could actually use have their WAV checked on disk.
pub fn pick_entries(
    history: &HistoryManager,
    entries: &[HistoryEntry],
    selection: ReplayBenchSelection,
) -> Vec<HistoryEntry> {
    let by_id: HashMap<i64, &HistoryEntry> = entries.iter().map(|e| (e.id, e)).collect();
    let candidates: Vec<Candidate<'_>> = entries
        .iter()
        .map(|entry| Candidate {
            id: entry.id,
            timestamp: entry.timestamp,
            saved: entry.saved,
            transcription_text: &entry.transcription_text,
        })
        .collect();
    select_entries(&candidates, selection, |c| {
        by_id
            .get(&c.id)
            .is_some_and(|entry| history.get_audio_file_path(&entry.file_name).is_file())
    })
    .into_iter()
    .filter_map(|id| by_id.get(&id).map(|entry| (*entry).clone()))
    .collect()
}

/// Read the WAV header only: duration and whether it is the 16 kHz mono
/// 16-bit PCM that Handy saves (anything else can't be replayed faithfully).
fn probe_wav(path: &std::path::Path) -> Option<f64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let valid = spec.sample_rate == 16_000
        && spec.channels == 1
        && spec.bits_per_sample == 16
        && spec.sample_format == hound::SampleFormat::Int;
    valid.then(|| f64::from(reader.duration()) / SAMPLE_RATE)
}

struct PlannedEntry {
    entry: ReplayBenchEntry,
    audio_path: std::path::PathBuf,
}

/// Validate a request and start the bench on a background thread. Results
/// arrive as [`ReplayBenchEvent`]s.
pub async fn start(app: &AppHandle, request: ReplayBenchRequest) -> Result<(), String> {
    let model_manager = app
        .try_state::<Arc<ModelManager>>()
        .ok_or("Model manager unavailable")?
        .inner()
        .clone();
    let history = app
        .try_state::<Arc<HistoryManager>>()
        .ok_or("History manager unavailable")?
        .inner()
        .clone();

    let mut models = Vec::new();
    for model_id in &request.model_ids {
        if models
            .iter()
            .any(|m: &ReplayBenchModel| &m.model_id == model_id)
        {
            continue;
        }
        let info = model_manager
            .get_model_info(model_id)
            .filter(|info| info.is_downloaded)
            .ok_or_else(|| format!("Model is not installed: {}", model_id))?;
        models.push(ReplayBenchModel {
            model_id: info.id,
            model_name: info.name,
        });
    }
    if models.is_empty() {
        return Err("no_models".to_string());
    }

    let all_entries = history
        .get_history_entries(None, None)
        .await
        .map_err(|e| e.to_string())?
        .entries;
    let selected = pick_entries(&history, &all_entries, request.selection);
    if selected.is_empty() {
        return Err("no_recordings".to_string());
    }

    if RUNNING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("already_running".to_string());
    }
    let guard = RunningGuard;
    CANCEL.store(false, Ordering::Release);

    let planned: Vec<PlannedEntry> = selected
        .iter()
        .map(|entry| {
            let audio_path = history.get_audio_file_path(&entry.file_name);
            let duration = probe_wav(&audio_path);
            PlannedEntry {
                entry: ReplayBenchEntry {
                    entry_id: entry.id,
                    timestamp: entry.timestamp,
                    reference_text: reference_text(
                        &entry.transcription_text,
                        entry.post_processed_text.as_deref(),
                        request.include_cleanup,
                    )
                    .to_string(),
                    duration_secs: duration.unwrap_or(0.0),
                    readable: duration.is_some(),
                },
                audio_path,
            }
        })
        .collect();

    let app = app.clone();
    let include_cleanup = request.include_cleanup;
    let spawned = thread::Builder::new()
        .name("replay-bench".to_string())
        .spawn(move || {
            lower_current_thread_priority();
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                run(&app, &model_manager, &planned, &models, include_cleanup)
            }));
            if outcome.is_err() {
                error!("Replay bench panicked; ending the run");
            }
            let was_cancelled = cancelled() || outcome.is_err();
            CANCEL.store(false, Ordering::Release);
            // Clear RUNNING before announcing the end, so a Run clicked the
            // moment the UI shows "finished" is not refused as already running.
            drop(guard);
            info!("Replay bench finished (cancelled: {})", was_cancelled);
            emit(
                &app,
                ReplayBenchEvent::Finished {
                    cancelled: was_cancelled,
                },
            );
        });
    // On spawn failure the closure (and the guard inside it) is dropped,
    // which clears RUNNING again.
    spawned.map(|_| ()).map_err(|e| e.to_string())
}

/// Best effort: run the bench below normal priority so a dictation that
/// starts mid-clip (load, transcribe and cleanup can't be interrupted) wins
/// the CPU. On Linux, threads the engine spawns from here inherit it; on
/// Windows they don't, which is why bench engines also get a CPU thread cap
/// (see `IsolatedEngine`). Cleanup runs off this thread (`run_bench_cleanup`)
/// so the shared async runtime never inherits the lowered priority.
fn lower_current_thread_priority() {
    #[cfg(windows)]
    {
        use windows::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
        };
        // SAFETY: GetCurrentThread returns a pseudo-handle that is always
        // valid for the calling thread and needs no closing.
        if let Err(e) =
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) }
        {
            warn!("Replay bench could not lower its thread priority: {}", e);
        }
    }
    #[cfg(target_os = "linux")]
    {
        // On Linux, PRIO_PROCESS with id 0 sets the calling thread's nice value.
        // SAFETY: plain syscall with no pointers.
        if unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10) } != 0 {
            warn!(
                "Replay bench could not lower its thread priority: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

fn emit(app: &AppHandle, event: ReplayBenchEvent) {
    if let Err(e) = event.emit(app) {
        warn!("Failed to emit replay bench event: {}", e);
    }
}

/// True while a dictation is recording or being processed.
fn dictation_active(app: &AppHandle) -> bool {
    let coordinator_busy = app
        .try_state::<TranscriptionCoordinator>()
        .is_some_and(|c| c.is_busy());
    let recording = app
        .try_state::<Arc<AudioRecordingManager>>()
        .is_some_and(|a| a.is_recording());
    coordinator_busy || recording
}

/// Block while a dictation is in progress so it gets the CPU/GPU. Returns
/// false if the run was stopped while waiting.
fn wait_for_dictation(app: &AppHandle, model: &ReplayBenchModel, index: u32, total: u32) -> bool {
    let mut paused = false;
    loop {
        if cancelled() {
            return false;
        }
        if !dictation_active(app) {
            return true;
        }
        if !paused {
            paused = true;
            emit(app, progress(model, index, total, ReplayBenchPhase::Paused));
        }
        thread::sleep(PAUSE_POLL_INTERVAL);
    }
}

fn progress(
    model: &ReplayBenchModel,
    entry_index: u32,
    entry_total: u32,
    phase: ReplayBenchPhase,
) -> ReplayBenchEvent {
    ReplayBenchEvent::Progress {
        model_id: model.model_id.clone(),
        model_name: model.model_name.clone(),
        entry_index,
        entry_total,
        phase,
    }
}

fn run(
    app: &AppHandle,
    model_manager: &ModelManager,
    entries: &[PlannedEntry],
    models: &[ReplayBenchModel],
    include_cleanup: bool,
) {
    let settings = get_settings(app);
    let total = u32::try_from(entries.len()).unwrap_or(u32::MAX);
    info!(
        "Replay bench started: {} recordings x {} models (cleanup: {})",
        entries.len(),
        models.len(),
        include_cleanup
    );
    emit(
        app,
        ReplayBenchEvent::Started {
            entries: entries.iter().map(|p| p.entry.clone()).collect(),
            models: models.to_vec(),
            include_cleanup,
        },
    );

    'models: for model in models {
        if !wait_for_dictation(app, model, 0, total) {
            break;
        }
        emit(app, progress(model, 0, total, ReplayBenchPhase::Loading));
        let load_start = Instant::now();
        let mut engine = match load_engine(app, model_manager, &model.model_id) {
            Ok(engine) => Some(engine),
            Err(e) => {
                warn!("Replay bench could not load {}: {}", model.model_id, e);
                emit(
                    app,
                    ReplayBenchEvent::ModelFinished {
                        summary: summarize(&model.model_id, None, Some(e.to_string()), &[]),
                    },
                );
                continue;
            }
        };
        let load_ms = elapsed_ms(load_start);

        let mut results = Vec::with_capacity(entries.len());
        for (index, planned) in entries.iter().enumerate() {
            let entry_index = u32::try_from(index + 1).unwrap_or(u32::MAX);
            if dictation_active(app) {
                // Free the bench model while the user dictates so the live
                // model has the memory (and VRAM) to itself, then reload.
                release_engine(&mut engine);
                if !wait_for_dictation(app, model, entry_index, total) {
                    emit_model_finished(app, model, load_ms, &results);
                    break 'models;
                }
            }
            if engine.is_none() {
                // Freed for a dictation (here or while an earlier entry
                // waited to run its cleanup): load it again.
                emit(
                    app,
                    progress(model, entry_index, total, ReplayBenchPhase::Loading),
                );
                match load_engine(app, model_manager, &model.model_id) {
                    Ok(loaded) => engine = Some(loaded),
                    Err(e) => {
                        warn!("Replay bench could not reload {}: {}", model.model_id, e);
                        emit_model_finished(app, model, load_ms, &results);
                        continue 'models;
                    }
                }
            }
            if cancelled() {
                release_engine(&mut engine);
                emit_model_finished(app, model, load_ms, &results);
                break 'models;
            }
            emit(
                app,
                progress(model, entry_index, total, ReplayBenchPhase::Transcribing),
            );
            let result = bench_entry(
                app,
                &mut engine,
                model_manager,
                &settings,
                planned,
                model,
                (entry_index, total),
                include_cleanup,
            );
            emit(
                app,
                ReplayBenchEvent::Result {
                    result: result.clone(),
                },
            );
            results.push(result);
        }

        // Free this model before the next one loads.
        release_engine(&mut engine);
        emit_model_finished(app, model, load_ms, &results);
    }
}

/// Drop the bench engine (if loaded) and hand its memory back to the OS.
fn release_engine(engine: &mut Option<IsolatedEngine>) {
    if engine.take().is_some() {
        crate::memory::trim_freed_memory();
    }
}

/// Load a bench engine with the current accelerator settings (the live
/// pipeline only re-applies them on its own loads).
fn load_engine(
    app: &AppHandle,
    model_manager: &ModelManager,
    model_id: &str,
) -> anyhow::Result<IsolatedEngine> {
    crate::managers::transcription::apply_accelerator_settings(app);
    IsolatedEngine::load(app, model_manager, model_id)
}

fn emit_model_finished(
    app: &AppHandle,
    model: &ReplayBenchModel,
    load_ms: f64,
    results: &[ReplayBenchResult],
) {
    emit(
        app,
        ReplayBenchEvent::ModelFinished {
            summary: summarize(&model.model_id, Some(load_ms), None, results),
        },
    );
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[allow(clippy::too_many_arguments)]
fn bench_entry(
    app: &AppHandle,
    engine: &mut Option<IsolatedEngine>,
    model_manager: &ModelManager,
    settings: &AppSettings,
    planned: &PlannedEntry,
    model: &ReplayBenchModel,
    (entry_index, total): (u32, u32),
    include_cleanup: bool,
) -> ReplayBenchResult {
    let mut result = ReplayBenchResult {
        entry_id: planned.entry.entry_id,
        model_id: model.model_id.clone(),
        failure: None,
        text: String::new(),
        cleanup_text: None,
        cleanup_failed: false,
        differs_pct: None,
        diff: Vec::new(),
        transcribe_ms: None,
        rtf: None,
        cleanup_ms: None,
    };

    let samples = match planned
        .entry
        .readable
        .then(|| crate::audio_toolkit::read_wav_samples(&planned.audio_path))
    {
        Some(Ok(samples)) if !samples.is_empty() => samples,
        _ => {
            result.failure = Some(ReplayBenchFailure::DecodeFailed);
            return result;
        }
    };
    let audio_secs = samples.len() as f64 / SAMPLE_RATE;

    let Some(loaded) = engine.as_mut() else {
        result.failure = Some(ReplayBenchFailure::TranscribeFailed);
        return result;
    };
    let start = Instant::now();
    let text = match loaded.transcribe(&samples, settings, model_manager) {
        Ok(text) => text,
        Err(e) => {
            warn!(
                "Replay bench: {} failed on entry {}: {}",
                model.model_id, planned.entry.entry_id, e
            );
            result.failure = Some(ReplayBenchFailure::TranscribeFailed);
            return result;
        }
    };
    let transcribe_ms = elapsed_ms(start);
    drop(samples);
    result.transcribe_ms = Some(transcribe_ms);
    result.rtf = real_time_factor(transcribe_ms, audio_secs);

    let mut compared = text.clone();
    // A dictation may have started during transcription. Cleanup can't be
    // interrupted once sent, and a single-slot local LLM would make the live
    // cleanup queue behind it, so wait for the dictation to finish first,
    // without holding the bench model (the next entry reloads it).
    if include_cleanup && dictation_active(app) {
        release_engine(engine);
    }
    if include_cleanup && wait_for_dictation(app, model, entry_index, total) {
        emit(
            app,
            progress(model, entry_index, total, ReplayBenchPhase::Cleanup),
        );
        let start = Instant::now();
        let outcome = run_bench_cleanup(settings, &text);
        match outcome {
            CleanupOutcome::Cleaned(cleaned) => {
                result.cleanup_ms = Some(elapsed_ms(start));
                compared = cleaned.clone();
                result.cleanup_text = Some(cleaned);
            }
            CleanupOutcome::Failed => {
                result.cleanup_ms = Some(elapsed_ms(start));
                result.cleanup_failed = true;
            }
            // Nothing configured (or blank text): not a failure, no timing.
            CleanupOutcome::Skipped => {}
        }
    }

    let diff = word_diff(&planned.entry.reference_text, &compared);
    result.differs_pct = Some(diff.differs_pct());
    result.diff = diff.tokens;
    result.text = text;
    result
}

/// Run the bench cleanup on a runtime worker rather than the bench thread.
/// The bench thread runs below normal priority, and on Linux any thread it
/// creates inherits that nice value; awaiting the cleanup here would let the
/// runtime's blocking pool grow niced threads that live dictations reuse.
/// No app context: the bench replays audio outside the app it was dictated
/// into, so it always measures the selected prompt.
fn run_bench_cleanup(settings: &AppSettings, text: &str) -> CleanupOutcome {
    let settings = settings.clone();
    let text = text.to_string();
    let task = tauri::async_runtime::spawn(async move {
        crate::actions::run_cleanup(
            &settings,
            &text,
            &crate::app_context::CleanupRequest::plain(),
            &RequestSent::detached(),
        )
        .await
    });
    match tauri::async_runtime::block_on(task) {
        Ok(outcome) => outcome,
        Err(e) => {
            warn!("Replay bench cleanup task failed: {}", e);
            CleanupOutcome::Failed
        }
    }
}

fn summarize(
    model_id: &str,
    load_ms: Option<f64>,
    load_error: Option<String>,
    results: &[ReplayBenchResult],
) -> ReplayBenchModelSummary {
    let transcribe: Vec<f64> = results.iter().filter_map(|r| r.transcribe_ms).collect();
    let rtf: Vec<f64> = results.iter().filter_map(|r| r.rtf).collect();
    let cleanup: Vec<f64> = results.iter().filter_map(|r| r.cleanup_ms).collect();
    let differs: Vec<f64> = results.iter().filter_map(|r| r.differs_pct).collect();
    let failed = results.iter().filter(|r| r.failure.is_some()).count();
    ReplayBenchModelSummary {
        model_id: model_id.to_string(),
        load_ms,
        load_error,
        transcribe_median_ms: median(&transcribe),
        transcribe_p90_ms: percentile(&transcribe, 90.0),
        rtf_median: median(&rtf),
        cleanup_median_ms: median(&cleanup),
        cleanup_p90_ms: percentile(&cleanup, 90.0),
        mean_differs_pct: mean(&differs),
        completed: u32::try_from(results.len() - failed).unwrap_or(u32::MAX),
        failed: u32::try_from(failed).unwrap_or(u32::MAX),
    }
}
