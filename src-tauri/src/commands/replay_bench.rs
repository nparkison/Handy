use crate::managers::history::HistoryManager;
use crate::replay_bench::{self, metrics::ReplayBenchSelection, ReplayBenchRequest};
use std::sync::Arc;
use tauri::{AppHandle, State};

#[tauri::command]
#[specta::specta]
pub async fn start_replay_bench(app: AppHandle, request: ReplayBenchRequest) -> Result<(), String> {
    replay_bench::start(&app, request).await
}

#[tauri::command]
#[specta::specta]
pub fn stop_replay_bench() {
    replay_bench::request_stop();
}

#[tauri::command]
#[specta::specta]
pub fn is_replay_bench_running() -> bool {
    replay_bench::is_running()
}

#[tauri::command]
#[specta::specta]
pub async fn count_replay_bench_recordings(
    history_manager: State<'_, Arc<HistoryManager>>,
    selection: ReplayBenchSelection,
) -> Result<u32, String> {
    let entries = history_manager
        .get_history_entries(None, None)
        .await
        .map_err(|e| e.to_string())?
        .entries;
    let picked = replay_bench::pick_entries(&history_manager, &entries, selection);
    Ok(u32::try_from(picked.len()).unwrap_or(u32::MAX))
}
