use anyhow::{anyhow, Result};
use chrono::{DateTime, Local, Utc};
use log::{debug, error, info};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::fs;
use std::path::PathBuf;
use tauri::AppHandle;
use tauri_specta::Event;

/// Database migrations for transcription history.
/// Each migration is applied in order. The library tracks which migrations
/// have been applied using SQLite's user_version pragma.
///
/// This list must stay identical to upstream Handy's (v0.9.8: four entries).
/// The fork shares `history.db` with upstream builds (same app identifier and
/// data dir), so a fork-only entry here would push `user_version` past what
/// upstream knows: an upstream build would then refuse to open the database
/// (`DatabaseTooFarAhead`, a panic at startup), and a future upstream
/// migration with the same number would be skipped. Fork-only columns are
/// added by [`ensure_fork_columns`] instead.
///
/// Note: For users upgrading from tauri-plugin-sql, migrate_from_tauri_plugin_sql()
/// converts the old _sqlx_migrations table tracking to the user_version pragma,
/// ensuring migrations don't re-run on existing databases.
static MIGRATIONS: &[M] = &[
    M::up(
        "CREATE TABLE IF NOT EXISTS transcription_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            saved BOOLEAN NOT NULL DEFAULT 0,
            title TEXT NOT NULL,
            transcription_text TEXT NOT NULL
        );",
    ),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_processed_text TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_prompt TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_requested BOOLEAN NOT NULL DEFAULT 0;"),
];

/// Fork-only columns, added idempotently after upstream's migrations without
/// touching `user_version`. Upstream builds ignore extra columns: their
/// INSERTs name columns explicitly and every one of these is nullable or has
/// a default.
const FORK_COLUMNS: &[(&str, &str)] = &[
    ("cleanup_state", "TEXT"),
    ("context_app", "TEXT"),
    ("context_title", "TEXT"),
    ("context_screenshot", "BOOLEAN NOT NULL DEFAULT 0"),
    // Executable/app identifier rules match on (e.g. `slack.exe`), so a retry
    // re-selects the same app rule. Local only: never shared or shown.
    ("context_process", "TEXT"),
];

/// Builds before this fix recorded the fork columns as migrations 5 (cleanup
/// state) and 6 (context). Such a database is upstream schema 4 plus fork
/// columns, so its `user_version` is put back to 4 to stay openable by
/// upstream builds.
fn reset_fork_schema_version(conn: &Connection) -> Result<()> {
    let upstream_version = MIGRATIONS.len() as i32;
    let version: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if !matches!(version, 5 | 6) {
        // Anything else is either current or belongs to a newer upstream;
        // leave the latter for rusqlite_migration to report.
        return Ok(());
    }
    let present = existing_columns(conn)?;
    // Migration 5 added cleanup_state; migration 6 the three context_*
    // columns. Columns added later (context_process) were never migrations.
    let fork_owned = FORK_COLUMNS
        .iter()
        .take(if version == 5 { 1 } else { 4 })
        .all(|(name, _)| present.iter().any(|c| c == name));
    if fork_owned {
        info!(
            "History database at fork schema version {}; resetting to upstream version {}",
            version, upstream_version
        );
        conn.pragma_update(None, "user_version", upstream_version)?;
    }
    Ok(())
}

fn existing_columns(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info('transcription_history')")?;
    let names = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(names)
}

/// Add any missing [`FORK_COLUMNS`]. Safe to run on every start.
fn ensure_fork_columns(conn: &Connection) -> Result<()> {
    let present = existing_columns(conn)?;
    for (name, definition) in FORK_COLUMNS {
        if !present.iter().any(|c| c == name) {
            conn.execute_batch(&format!(
                "ALTER TABLE transcription_history ADD COLUMN {name} {definition};"
            ))?;
            debug!("Added fork history column {}", name);
        }
    }
    Ok(())
}

/// Bring a database to the current schema: upstream migrations, then the
/// fork's columns. See [`MIGRATIONS`] for why the two are kept apart.
fn migrate_schema(conn: &mut Connection) -> Result<()> {
    reset_fork_schema_version(conn)?;
    let migrations = Migrations::new(MIGRATIONS.to_vec());
    // Validate migrations in debug builds
    #[cfg(debug_assertions)]
    migrations.validate().expect("Invalid migrations");
    migrations.to_latest(conn)?;
    ensure_fork_columns(conn)
}

/// Registers `handy_lower(text)`, a Unicode lowercase for search. SQLite's
/// own `lower()` and `LIKE` only fold ASCII, while History's highlighter
/// folds all of Unicode.
fn register_search_functions(conn: &Connection) -> rusqlite::Result<()> {
    use rusqlite::functions::FunctionFlags;
    conn.create_scalar_function(
        "handy_lower",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            Ok(match ctx.get_raw(0) {
                rusqlite::types::ValueRef::Text(bytes) => {
                    Some(String::from_utf8_lossy(bytes).to_lowercase())
                }
                _ => None,
            })
        },
    )
}

/// Columns read by [`HistoryManager::map_history_entry`], in SELECT order.
macro_rules! entry_columns {
    () => {
        "id, file_name, timestamp, saved, title, transcription_text, post_processed_text, \
         post_process_prompt, post_process_requested, cleanup_state, context_app, \
         context_title, context_screenshot"
    };
}

/// What app context was sent with an entry's cleanup request: never the
/// screenshot itself, only whether one was shared.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct HistoryContext {
    pub app: Option<String>,
    pub title: Option<String>,
    pub screenshot: bool,
}

impl HistoryContext {
    /// `None` when nothing was shared.
    pub fn into_option(self) -> Option<Self> {
        (self.app.is_some() || self.title.is_some() || self.screenshot).then_some(self)
    }
}

/// A deadline-missed cleanup that arrived: its text, the prompt it used, and
/// whether the screenshot shaped it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LateCleanup {
    pub text: String,
    pub prompt: Option<String>,
    pub screenshot: bool,
}

/// Where a deadline-missed cleanup stands. `None` on an entry means cleanup
/// either finished in time, failed, or never ran (see `post_process_requested`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum CleanupState {
    /// Missed the deadline; the original was pasted and the request is still running.
    Pending,
    /// The cleaned-up text arrived after the deadline and was saved here.
    Late,
}

impl CleanupState {
    fn as_db(self) -> &'static str {
        match self {
            CleanupState::Pending => "pending",
            CleanupState::Late => "late",
        }
    }

    fn from_db(value: Option<String>) -> Option<Self> {
        match value.as_deref() {
            Some("pending") => Some(CleanupState::Pending),
            Some("late") => Some(CleanupState::Late),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct PaginatedHistory {
    pub entries: Vec<HistoryEntry>,
    pub has_more: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
#[serde(tag = "action")]
pub enum HistoryUpdatePayload {
    #[serde(rename = "added")]
    Added { entry: HistoryEntry },
    #[serde(rename = "updated")]
    Updated { entry: HistoryEntry },
    #[serde(rename = "deleted")]
    Deleted { id: i64 },
    #[serde(rename = "toggled")]
    Toggled { id: i64 },
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct HistoryEntry {
    pub id: i64,
    pub file_name: String,
    pub timestamp: i64,
    pub saved: bool,
    pub title: String,
    pub transcription_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    pub post_process_requested: bool,
    pub cleanup_state: Option<CleanupState>,
    /// App context sent with the cleanup request (`None` when none was sent).
    pub context: Option<HistoryContext>,
}

pub struct HistoryManager {
    app_handle: AppHandle,
    recordings_dir: PathBuf,
    db_path: PathBuf,
}

impl HistoryManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        // Create recordings directory in app data dir
        let app_data_dir = crate::portable::app_data_dir(app_handle)?;
        let recordings_dir = app_data_dir.join("recordings");
        let db_path = app_data_dir.join("history.db");

        // Ensure recordings directory exists
        if !recordings_dir.exists() {
            fs::create_dir_all(&recordings_dir)?;
            debug!("Created recordings directory: {:?}", recordings_dir);
        }

        let manager = Self {
            app_handle: app_handle.clone(),
            recordings_dir,
            db_path,
        };

        // Initialize database and run migrations synchronously
        manager.init_database()?;

        Ok(manager)
    }

    fn init_database(&self) -> Result<()> {
        info!("Initializing database at {:?}", self.db_path);

        let mut conn = Connection::open(&self.db_path)?;

        // Handle migration from tauri-plugin-sql to rusqlite_migration
        // tauri-plugin-sql used _sqlx_migrations table, rusqlite_migration uses user_version pragma
        self.migrate_from_tauri_plugin_sql(&conn)?;

        // Get current version before migration
        let version_before: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
        debug!("Database version before migration: {}", version_before);

        // Apply any pending upstream migrations, then the fork's columns.
        migrate_schema(&mut conn)?;

        // A cleanup still pending from a previous run will never arrive.
        let stale = conn.execute(
            "UPDATE transcription_history SET cleanup_state = NULL WHERE cleanup_state = 'pending'",
            [],
        )?;
        if stale > 0 {
            debug!("Cleared {} stale pending cleanups", stale);
        }

        // Get version after migration
        let version_after: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version_after > version_before {
            info!(
                "Database migrated from version {} to {}",
                version_before, version_after
            );
        } else {
            debug!("Database already at latest version {}", version_after);
        }

        Ok(())
    }

    /// Migrate from tauri-plugin-sql's migration tracking to rusqlite_migration's.
    /// tauri-plugin-sql used a _sqlx_migrations table, while rusqlite_migration uses
    /// SQLite's user_version pragma. This function checks if the old system was in use
    /// and sets the user_version accordingly so migrations don't re-run.
    fn migrate_from_tauri_plugin_sql(&self, conn: &Connection) -> Result<()> {
        // Check if the old _sqlx_migrations table exists
        let has_sqlx_migrations: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='_sqlx_migrations'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !has_sqlx_migrations {
            return Ok(());
        }

        // Check current user_version
        let current_version: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version > 0 {
            // Already migrated to rusqlite_migration system
            return Ok(());
        }

        // Get the highest version from the old migrations table
        let old_version: i32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM _sqlx_migrations WHERE success = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if old_version > 0 {
            info!(
                "Migrating from tauri-plugin-sql (version {}) to rusqlite_migration",
                old_version
            );

            // Set user_version to match the old migration state
            conn.pragma_update(None, "user_version", old_version)?;

            // Optionally drop the old migrations table (keeping it doesn't hurt)
            // conn.execute("DROP TABLE IF EXISTS _sqlx_migrations", [])?;

            info!(
                "Migration tracking converted: user_version set to {}",
                old_version
            );
        }

        Ok(())
    }

    fn get_connection(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    fn map_history_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<HistoryEntry> {
        Ok(HistoryEntry {
            id: row.get("id")?,
            file_name: row.get("file_name")?,
            timestamp: row.get("timestamp")?,
            saved: row.get("saved")?,
            title: row.get("title")?,
            transcription_text: row.get("transcription_text")?,
            post_processed_text: row.get("post_processed_text")?,
            post_process_prompt: row.get("post_process_prompt")?,
            post_process_requested: row.get("post_process_requested")?,
            cleanup_state: CleanupState::from_db(row.get("cleanup_state")?),
            context: HistoryContext {
                app: row.get("context_app")?,
                title: row.get("context_title")?,
                screenshot: row.get("context_screenshot")?,
            }
            .into_option(),
        })
    }

    pub fn recordings_dir(&self) -> &std::path::Path {
        &self.recordings_dir
    }

    /// Save a new history entry to the database.
    /// The WAV file should already have been written to the recordings directory.
    #[allow(clippy::too_many_arguments)]
    pub fn save_entry(
        &self,
        file_name: String,
        transcription_text: String,
        post_process_requested: bool,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        cleanup_state: Option<CleanupState>,
        context: HistoryContext,
        match_process: Option<String>,
    ) -> Result<HistoryEntry> {
        let timestamp = Utc::now().timestamp();
        let title = self.format_timestamp_title(timestamp);

        let conn = self.get_connection()?;
        conn.execute(
            "INSERT INTO transcription_history (
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                post_process_requested,
                cleanup_state,
                context_app,
                context_title,
                context_screenshot,
                context_process
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                &file_name,
                timestamp,
                false,
                &title,
                &transcription_text,
                &post_processed_text,
                &post_process_prompt,
                post_process_requested,
                cleanup_state.map(CleanupState::as_db),
                &context.app,
                &context.title,
                context.screenshot,
                &match_process,
            ],
        )?;

        let entry = HistoryEntry {
            id: conn.last_insert_rowid(),
            file_name,
            timestamp,
            saved: false,
            title,
            transcription_text,
            post_processed_text,
            post_process_prompt,
            post_process_requested,
            cleanup_state,
            context: context.into_option(),
        };

        debug!("Saved history entry with id {}", entry.id);

        self.cleanup_old_entries()?;

        // Emit typed event for real-time frontend updates
        if let Err(e) = (HistoryUpdatePayload::Added {
            entry: entry.clone(),
        })
        .emit(&self.app_handle)
        {
            error!("Failed to emit history-updated event: {}", e);
        }
        crate::tray::refresh_recent_dictations(&self.app_handle);

        Ok(entry)
    }

    /// Update an existing history entry with new transcription results (used by retry).
    /// `post_process_requested` records whether a cleanup request was
    /// attempted; `context` is what that request shared, or `None` to keep
    /// the stored context (no request went out, so nothing new was shared).
    pub fn update_transcription(
        &self,
        id: i64,
        transcription_text: String,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        post_process_requested: bool,
        context: Option<HistoryContext>,
    ) -> Result<HistoryEntry> {
        let conn = self.get_connection()?;
        let updated = Self::update_transcription_with_conn(
            &conn,
            id,
            transcription_text,
            post_processed_text,
            post_process_prompt,
            post_process_requested,
            context,
        )?;
        if updated == 0 {
            return Err(anyhow!("History entry {} not found", id));
        }

        let entry = conn.query_row(
            concat!(
                "SELECT ",
                entry_columns!(),
                " FROM transcription_history WHERE id = ?1"
            ),
            params![id],
            Self::map_history_entry,
        )?;

        debug!("Updated transcription for history entry {}", id);

        if let Err(e) = (HistoryUpdatePayload::Updated {
            entry: entry.clone(),
        })
        .emit(&self.app_handle)
        {
            error!("Failed to emit history-updated event: {}", e);
        }
        crate::tray::refresh_recent_dictations(&self.app_handle);

        Ok(entry)
    }

    fn update_transcription_with_conn(
        conn: &Connection,
        id: i64,
        transcription_text: String,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        post_process_requested: bool,
        context: Option<HistoryContext>,
    ) -> Result<usize> {
        let keep_context = context.is_none();
        let context = context.unwrap_or_default();
        Ok(conn.execute(
            "UPDATE transcription_history
             SET transcription_text = ?1,
                 post_processed_text = ?2,
                 post_process_prompt = ?3,
                 post_process_requested = ?8,
                 cleanup_state = NULL,
                 context_app = CASE WHEN ?9 THEN context_app ELSE ?4 END,
                 context_title = CASE WHEN ?9 THEN context_title ELSE ?5 END,
                 context_screenshot = CASE WHEN ?9 THEN context_screenshot ELSE ?6 END
             WHERE id = ?7",
            params![
                transcription_text,
                post_processed_text,
                post_process_prompt,
                context.app,
                context.title,
                context.screenshot,
                id,
                post_process_requested,
                keep_context
            ],
        )?)
    }

    /// Resolve a deadline-missed cleanup: store the late cleaned-up text
    /// (`Some`) or record that it never arrived (`None`, shown as "Cleanup
    /// failed"). Only touches an entry that is still pending, so a re-transcribe
    /// in the meantime is never overwritten. Returns the updated entry, or
    /// `None` when it was deleted or no longer pending.
    pub fn resolve_late_cleanup(
        &self,
        id: i64,
        cleaned: Option<LateCleanup>,
    ) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        let updated = Self::resolve_late_cleanup_with_conn(&conn, id, cleaned)?;
        if let Some(entry) = &updated {
            if let Err(e) = (HistoryUpdatePayload::Updated {
                entry: entry.clone(),
            })
            .emit(&self.app_handle)
            {
                error!("Failed to emit history-updated event: {}", e);
            }
            crate::tray::refresh_recent_dictations(&self.app_handle);
        }
        Ok(updated)
    }

    fn resolve_late_cleanup_with_conn(
        conn: &Connection,
        id: i64,
        cleaned: Option<LateCleanup>,
    ) -> Result<Option<HistoryEntry>> {
        let changed = match cleaned {
            Some(LateCleanup {
                text,
                prompt,
                screenshot,
            }) => conn.execute(
                "UPDATE transcription_history
                 SET post_processed_text = ?1, post_process_prompt = ?2, cleanup_state = ?3,
                     context_screenshot = (context_screenshot OR ?6)
                 WHERE id = ?4 AND cleanup_state = ?5",
                params![
                    text,
                    prompt,
                    CleanupState::Late.as_db(),
                    id,
                    CleanupState::Pending.as_db(),
                    screenshot
                ],
            )?,
            None => conn.execute(
                "UPDATE transcription_history SET cleanup_state = NULL
                 WHERE id = ?1 AND cleanup_state = ?2",
                params![id, CleanupState::Pending.as_db()],
            )?,
        };
        if changed == 0 {
            return Ok(None);
        }
        Ok(conn
            .query_row(
                concat!(
                    "SELECT ",
                    entry_columns!(),
                    " FROM transcription_history WHERE id = ?1"
                ),
                params![id],
                Self::map_history_entry,
            )
            .optional()?)
    }

    pub fn cleanup_old_entries(&self) -> Result<()> {
        let retention_period = crate::settings::get_recording_retention_period(&self.app_handle);

        match retention_period {
            crate::settings::RecordingRetentionPeriod::Never => {
                // Don't delete anything
                Ok(())
            }
            crate::settings::RecordingRetentionPeriod::PreserveLimit => {
                // Use the old count-based logic with history_limit
                let limit = crate::settings::get_history_limit(&self.app_handle);
                self.cleanup_by_count(limit)
            }
            _ => {
                // Use time-based logic
                self.cleanup_by_time(retention_period)
            }
        }
    }

    fn delete_entries_and_files(&self, entries: &[(i64, String)]) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        let conn = self.get_connection()?;
        let mut deleted_count = 0;

        for (id, file_name) in entries {
            // Delete database entry
            conn.execute(
                "DELETE FROM transcription_history WHERE id = ?1",
                params![id],
            )?;

            // Delete WAV file
            let file_path = self.recordings_dir.join(file_name);
            if file_path.exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to delete WAV file {}: {}", file_name, e);
                } else {
                    debug!("Deleted old WAV file: {}", file_name);
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }

    fn cleanup_by_count(&self, limit: usize) -> Result<()> {
        let conn = self.get_connection()?;

        // Unsaved entries, newest first. `id` is the insertion order and never
        // ties; `timestamp` has one-second resolution and follows the wall clock.
        let mut stmt = conn.prepare(
            "SELECT id, file_name FROM transcription_history WHERE saved = 0 ORDER BY id DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
        })?;

        let mut entries: Vec<(i64, String)> = Vec::new();
        for row in rows {
            entries.push(row?);
        }

        if entries.len() > limit {
            let entries_to_delete = &entries[limit..];
            let deleted_count = self.delete_entries_and_files(entries_to_delete)?;

            if deleted_count > 0 {
                debug!("Cleaned up {} old history entries by count", deleted_count);
            }
        }

        Ok(())
    }

    fn cleanup_by_time(
        &self,
        retention_period: crate::settings::RecordingRetentionPeriod,
    ) -> Result<()> {
        let conn = self.get_connection()?;

        // Calculate cutoff timestamp (current time minus retention period)
        let now = Utc::now().timestamp();
        let cutoff_timestamp = match retention_period {
            crate::settings::RecordingRetentionPeriod::Days3 => now - (3 * 24 * 60 * 60), // 3 days in seconds
            crate::settings::RecordingRetentionPeriod::Weeks2 => now - (2 * 7 * 24 * 60 * 60), // 2 weeks in seconds
            crate::settings::RecordingRetentionPeriod::Months3 => now - (3 * 30 * 24 * 60 * 60), // 3 months in seconds (approximate)
            _ => unreachable!("Should not reach here"),
        };

        // Get all unsaved entries older than the cutoff timestamp
        let mut stmt = conn.prepare(
            "SELECT id, file_name FROM transcription_history WHERE saved = 0 AND timestamp < ?1",
        )?;

        let rows = stmt.query_map(params![cutoff_timestamp], |row| {
            Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
        })?;

        let mut entries_to_delete: Vec<(i64, String)> = Vec::new();
        for row in rows {
            entries_to_delete.push(row?);
        }

        let deleted_count = self.delete_entries_and_files(&entries_to_delete)?;

        if deleted_count > 0 {
            debug!(
                "Cleaned up {} old history entries based on retention period",
                deleted_count
            );
        }

        Ok(())
    }

    pub async fn get_history_entries(
        &self,
        cursor: Option<i64>,
        limit: Option<usize>,
    ) -> Result<PaginatedHistory> {
        let conn = self.get_connection()?;
        Self::query_history_page(&conn, cursor, limit, None)
    }

    /// Page through entries whose raw or polished text contains `query`,
    /// case-insensitively across Unicode (é/É, Cyrillic, ẞ/ß). Searches every
    /// retained entry, not just the page the UI has loaded.
    pub async fn search_history_entries(
        &self,
        query: &str,
        cursor: Option<i64>,
        limit: Option<usize>,
    ) -> Result<PaginatedHistory> {
        let conn = self.get_connection()?;
        let query = query.trim();
        let search = (!query.is_empty()).then_some(query);
        Self::query_history_page(&conn, cursor, limit, search)
    }

    /// Shared cursor-paginated query. `cursor` is the id of the last entry of
    /// the previous page (entries are returned newest-first by id, which is
    /// unique, so pages never skip or repeat entries that share a timestamp),
    /// `limit` is capped at 100, and `search` filters on both text columns.
    ///
    /// Matching lowercases both sides with Rust's Unicode `to_lowercase` (the
    /// same mapping as the frontend's `toLowerCase`) and uses `instr`, so user
    /// input never acts as a `LIKE` wildcard. `ß` matches `ß`/`ẞ` but not `ss`
    /// (lowercasing, not full case folding).
    fn query_history_page(
        conn: &Connection,
        cursor: Option<i64>,
        limit: Option<usize>,
        search: Option<&str>,
    ) -> Result<PaginatedHistory> {
        let limit = limit.map(|l| l.min(100));
        // Fetch one extra row to learn whether another page exists.
        // SQLite treats a negative LIMIT as "no limit".
        let fetch_count: i64 = limit.map_or(-1, |lim| lim as i64 + 1);
        let needle = search.map(str::to_lowercase);
        // The statement references the function even when not searching.
        register_search_functions(conn)?;

        let mut stmt = conn.prepare(concat!(
            "SELECT ",
            entry_columns!(),
            " FROM transcription_history
             WHERE (?1 IS NULL OR id < ?1)
               AND (?2 IS NULL
                    OR instr(handy_lower(transcription_text), ?2) > 0
                    OR instr(handy_lower(COALESCE(post_processed_text, '')), ?2) > 0)
             ORDER BY id DESC
             LIMIT ?3"
        ))?;
        let mut entries = stmt
            .query_map(
                params![cursor, needle, fetch_count],
                Self::map_history_entry,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let has_more = limit.is_some_and(|lim| entries.len() > lim);
        if has_more {
            entries.pop();
        }

        Ok(PaginatedHistory { entries, has_more })
    }

    /// Distinct app names recorded as cleanup context, most recently used
    /// first. Feeds the "Add rule" picker.
    pub fn get_recent_context_apps(&self, limit: usize) -> Result<Vec<String>> {
        let conn = self.get_connection()?;
        Self::get_recent_context_apps_with_conn(&conn, limit)
    }

    fn get_recent_context_apps_with_conn(conn: &Connection, limit: usize) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT context_app FROM transcription_history
             WHERE context_app IS NOT NULL AND context_app != ''
             GROUP BY context_app
             ORDER BY MAX(id) DESC
             LIMIT ?1",
        )?;
        let apps = stmt
            .query_map(params![limit as i64], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(apps)
    }

    /// The newest `limit` entries that have transcription text (failed
    /// recordings are skipped), newest first. Used by the tray's recent list.
    pub fn get_recent_completed_entries(&self, limit: usize) -> Result<Vec<HistoryEntry>> {
        let conn = self.get_connection()?;
        Self::get_recent_completed_entries_with_conn(&conn, limit)
    }

    fn get_recent_completed_entries_with_conn(
        conn: &Connection,
        limit: usize,
    ) -> Result<Vec<HistoryEntry>> {
        let mut stmt = conn.prepare(concat!(
            "SELECT ",
            entry_columns!(),
            " FROM transcription_history
             WHERE transcription_text != ''
             ORDER BY id DESC
             LIMIT ?1"
        ))?;
        let entries = stmt
            .query_map(params![limit as i64], Self::map_history_entry)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(entries)
    }

    #[cfg(test)]
    fn get_latest_entry_with_conn(conn: &Connection) -> Result<Option<HistoryEntry>> {
        let mut stmt = conn.prepare(concat!(
            "SELECT ",
            entry_columns!(),
            " FROM transcription_history
             ORDER BY id DESC
             LIMIT 1"
        ))?;

        let entry = stmt.query_row([], Self::map_history_entry).optional()?;
        Ok(entry)
    }

    /// Get the latest entry with non-empty transcription text.
    pub fn get_latest_completed_entry(&self) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        Self::get_latest_completed_entry_with_conn(&conn)
    }

    fn get_latest_completed_entry_with_conn(conn: &Connection) -> Result<Option<HistoryEntry>> {
        let mut stmt = conn.prepare(concat!(
            "SELECT ",
            entry_columns!(),
            " FROM transcription_history
             WHERE transcription_text != ''
             ORDER BY id DESC
             LIMIT 1"
        ))?;

        let entry = stmt.query_row([], Self::map_history_entry).optional()?;
        Ok(entry)
    }

    pub async fn toggle_saved_status(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;

        // Get current saved status
        let current_saved: bool = conn.query_row(
            "SELECT saved FROM transcription_history WHERE id = ?1",
            params![id],
            |row| row.get("saved"),
        )?;

        let new_saved = !current_saved;

        conn.execute(
            "UPDATE transcription_history SET saved = ?1 WHERE id = ?2",
            params![new_saved, id],
        )?;

        debug!("Toggled saved status for entry {}: {}", id, new_saved);

        // Emit history updated event
        if let Err(e) = (HistoryUpdatePayload::Toggled { id }).emit(&self.app_handle) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    pub fn get_audio_file_path(&self, file_name: &str) -> PathBuf {
        self.recordings_dir.join(file_name)
    }

    pub async fn get_entry_by_id(&self, id: i64) -> Result<Option<HistoryEntry>> {
        self.find_entry(id)
    }

    /// The app identifier rules matched on when entry `id` was dictated
    /// (see `context_process`), for re-selecting the app rule on retry.
    pub fn find_match_process(&self, id: i64) -> Result<Option<String>> {
        let conn = self.get_connection()?;
        Self::find_match_process_with_conn(&conn, id)
    }

    fn find_match_process_with_conn(conn: &Connection, id: i64) -> Result<Option<String>> {
        Ok(conn
            .query_row(
                "SELECT context_process FROM transcription_history WHERE id = ?1",
                [id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten()
            .filter(|p| !p.trim().is_empty()))
    }

    /// Synchronous lookup by id, for callers outside an async context (tray).
    pub fn find_entry(&self, id: i64) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(concat!(
            "SELECT ",
            entry_columns!(),
            " FROM transcription_history
             WHERE id = ?1"
        ))?;

        let entry = stmt.query_row([id], Self::map_history_entry).optional()?;

        Ok(entry)
    }

    pub async fn delete_entry(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;

        // Get the entry to find the file name
        if let Some(entry) = self.get_entry_by_id(id).await? {
            // Delete the audio file first
            let file_path = self.get_audio_file_path(&entry.file_name);
            if file_path.exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to delete audio file {}: {}", entry.file_name, e);
                    // Continue with database deletion even if file deletion fails
                }
            }
        }

        // Delete from database
        conn.execute(
            "DELETE FROM transcription_history WHERE id = ?1",
            params![id],
        )?;

        debug!("Deleted history entry with id: {}", id);

        // Emit history updated event
        if let Err(e) = (HistoryUpdatePayload::Deleted { id }).emit(&self.app_handle) {
            error!("Failed to emit history-updated event: {}", e);
        }
        crate::tray::refresh_recent_dictations(&self.app_handle);

        Ok(())
    }

    fn format_timestamp_title(&self, timestamp: i64) -> String {
        if let Some(utc_datetime) = DateTime::from_timestamp(timestamp, 0) {
            // Convert UTC to local timezone
            let local_datetime = utc_datetime.with_timezone(&Local);
            local_datetime.format("%B %e, %Y - %l:%M%p").to_string()
        } else {
            format!("Recording {}", timestamp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};

    fn setup_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch(
            "CREATE TABLE transcription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                saved BOOLEAN NOT NULL DEFAULT 0,
                title TEXT NOT NULL,
                transcription_text TEXT NOT NULL,
                post_processed_text TEXT,
                post_process_prompt TEXT,
                post_process_requested BOOLEAN NOT NULL DEFAULT 0,
                cleanup_state TEXT,
                context_app TEXT,
                context_title TEXT,
                context_screenshot BOOLEAN NOT NULL DEFAULT 0,
                context_process TEXT
            );",
        )
        .expect("create transcription_history table");
        conn
    }

    fn insert_entry(conn: &Connection, timestamp: i64, text: &str, post_processed: Option<&str>) {
        conn.execute(
            "INSERT INTO transcription_history (
                file_name,
                timestamp,
                saved,
                title,
                transcription_text,
                post_processed_text,
                post_process_prompt,
                post_process_requested
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                format!("handy-{}.wav", timestamp),
                timestamp,
                false,
                format!("Recording {}", timestamp),
                text,
                post_processed,
                Option::<String>::None,
                false,
            ],
        )
        .expect("insert history entry");
    }

    #[test]
    fn get_latest_entry_returns_none_when_empty() {
        let conn = setup_conn();
        let entry = HistoryManager::get_latest_entry_with_conn(&conn).expect("fetch latest entry");
        assert!(entry.is_none());
    }

    #[test]
    fn get_latest_entry_returns_newest_entry() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "first", None);
        insert_entry(&conn, 200, "second", Some("processed"));

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch latest entry")
            .expect("entry exists");

        assert_eq!(entry.timestamp, 200);
        assert_eq!(entry.transcription_text, "second");
        assert_eq!(entry.post_processed_text.as_deref(), Some("processed"));
    }

    #[test]
    fn get_latest_completed_entry_skips_empty_entries() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "completed", None);
        insert_entry(&conn, 200, "", None);

        let entry = HistoryManager::get_latest_completed_entry_with_conn(&conn)
            .expect("fetch latest completed entry")
            .expect("completed entry exists");

        assert_eq!(entry.timestamp, 100);
        assert_eq!(entry.transcription_text, "completed");
    }

    fn page_texts(page: &PaginatedHistory) -> Vec<&str> {
        page.entries
            .iter()
            .map(|e| e.transcription_text.as_str())
            .collect()
    }

    #[test]
    fn query_page_without_search_paginates_newest_first() {
        let conn = setup_conn();
        for i in 1..=5 {
            insert_entry(&conn, i * 100, &format!("entry {i}"), None);
        }

        let first = HistoryManager::query_history_page(&conn, None, Some(2), None).unwrap();
        assert_eq!(page_texts(&first), vec!["entry 5", "entry 4"]);
        assert!(first.has_more);

        let cursor = first.entries.last().map(|e| e.id);
        let second = HistoryManager::query_history_page(&conn, cursor, Some(2), None).unwrap();
        assert_eq!(page_texts(&second), vec!["entry 3", "entry 2"]);
        assert!(second.has_more);

        let cursor = second.entries.last().map(|e| e.id);
        let last = HistoryManager::query_history_page(&conn, cursor, Some(2), None).unwrap();
        assert_eq!(page_texts(&last), vec!["entry 1"]);
        assert!(!last.has_more);

        let all = HistoryManager::query_history_page(&conn, None, None, None).unwrap();
        assert_eq!(all.entries.len(), 5);
        assert!(!all.has_more);
    }

    #[test]
    fn search_matches_raw_and_polished_text_case_insensitively() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "meeting notes for Tuesday", None);
        insert_entry(
            &conn,
            200,
            "uh so the budget",
            Some("The Budget is approved."),
        );
        insert_entry(&conn, 300, "unrelated", Some("Also unrelated."));

        let page =
            HistoryManager::query_history_page(&conn, None, Some(30), Some("BUDGET")).unwrap();
        assert_eq!(page_texts(&page), vec!["uh so the budget"]);

        let page =
            HistoryManager::query_history_page(&conn, None, Some(30), Some("approved")).unwrap();
        assert_eq!(page_texts(&page), vec!["uh so the budget"]);

        let page =
            HistoryManager::query_history_page(&conn, None, Some(30), Some("tuesday")).unwrap();
        assert_eq!(page_texts(&page), vec!["meeting notes for Tuesday"]);

        let page =
            HistoryManager::query_history_page(&conn, None, Some(30), Some("nothing")).unwrap();
        assert!(page.entries.is_empty());
        assert!(!page.has_more);
    }

    #[test]
    fn search_treats_wildcards_and_quotes_literally() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "growth was 50% this year", None);
        insert_entry(&conn, 200, "growth was 500 units", None);
        insert_entry(&conn, 300, "snake_case name", None);
        insert_entry(&conn, 400, "snakeXcase name", None);
        insert_entry(&conn, 500, "it's Bobby'); DROP TABLE x;--", None);

        let page = HistoryManager::query_history_page(&conn, None, None, Some("50%")).unwrap();
        assert_eq!(page_texts(&page), vec!["growth was 50% this year"]);

        let page =
            HistoryManager::query_history_page(&conn, None, None, Some("snake_case")).unwrap();
        assert_eq!(page_texts(&page), vec!["snake_case name"]);

        let page = HistoryManager::query_history_page(&conn, None, None, Some("'); DROP")).unwrap();
        assert_eq!(page.entries.len(), 1);
        let all = HistoryManager::query_history_page(&conn, None, None, None).unwrap();
        assert_eq!(all.entries.len(), 5);
    }

    #[test]
    fn search_paginates_over_matches_only() {
        let conn = setup_conn();
        for i in 1..=6 {
            let text = if i % 2 == 0 {
                format!("match {i}")
            } else {
                format!("other {i}")
            };
            insert_entry(&conn, i * 100, &text, None);
        }

        let first =
            HistoryManager::query_history_page(&conn, None, Some(2), Some("match")).unwrap();
        assert_eq!(page_texts(&first), vec!["match 6", "match 4"]);
        assert!(first.has_more);

        let cursor = first.entries.last().map(|e| e.id);
        let second =
            HistoryManager::query_history_page(&conn, cursor, Some(2), Some("match")).unwrap();
        assert_eq!(page_texts(&second), vec!["match 2"]);
        assert!(!second.has_more);
    }

    #[test]
    fn recent_completed_entries_skip_failed_and_respect_limit() {
        let conn = setup_conn();
        for i in 1..=7 {
            insert_entry(&conn, i * 100, &format!("entry {i}"), None);
        }
        insert_entry(&conn, 800, "", None);

        let recent = HistoryManager::get_recent_completed_entries_with_conn(&conn, 5).unwrap();
        let texts: Vec<&str> = recent
            .iter()
            .map(|e| e.transcription_text.as_str())
            .collect();
        assert_eq!(
            texts,
            vec!["entry 7", "entry 6", "entry 5", "entry 4", "entry 3"]
        );
    }

    fn mark_pending(conn: &Connection, id: i64) {
        conn.execute(
            "UPDATE transcription_history SET cleanup_state = 'pending', post_process_requested = 1 WHERE id = ?1",
            params![id],
        )
        .expect("mark pending");
    }

    fn late(text: &str, prompt: Option<&str>, screenshot: bool) -> LateCleanup {
        LateCleanup {
            text: text.to_string(),
            prompt: prompt.map(str::to_string),
            screenshot,
        }
    }

    #[test]
    fn late_cleanup_saves_text_and_marks_late() {
        let conn = setup_conn();
        insert_entry(&conn, 1, "um hello", None);
        mark_pending(&conn, 1);

        let entry = HistoryManager::resolve_late_cleanup_with_conn(
            &conn,
            1,
            Some(late("Hello.", Some("prompt"), true)),
        )
        .expect("resolve")
        .expect("entry was pending");
        assert_eq!(entry.post_processed_text.as_deref(), Some("Hello."));
        assert_eq!(entry.post_process_prompt.as_deref(), Some("prompt"));
        assert_eq!(entry.cleanup_state, Some(CleanupState::Late));
        // The screenshot shaped the late result.
        assert!(entry.context.is_some_and(|c| c.screenshot));

        // Already resolved: a second result is ignored.
        let again = HistoryManager::resolve_late_cleanup_with_conn(
            &conn,
            1,
            Some(late("Other.", None, false)),
        )
        .expect("resolve");
        assert!(again.is_none());
    }

    #[test]
    fn late_cleanup_failure_clears_pending_state() {
        let conn = setup_conn();
        insert_entry(&conn, 1, "um hello", None);
        mark_pending(&conn, 1);
        let entry = HistoryManager::resolve_late_cleanup_with_conn(&conn, 1, None)
            .expect("resolve")
            .expect("entry was pending");
        assert_eq!(entry.cleanup_state, None);
        assert_eq!(entry.post_processed_text, None);
        assert!(entry.post_process_requested);
    }

    #[test]
    fn late_cleanup_skips_entries_that_are_not_pending() {
        let conn = setup_conn();
        insert_entry(&conn, 1, "um hello", Some("Hello."));
        let result = HistoryManager::resolve_late_cleanup_with_conn(
            &conn,
            1,
            Some(late("Late.", None, false)),
        )
        .expect("resolve");
        assert!(result.is_none());
        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert_eq!(entry.post_processed_text.as_deref(), Some("Hello."));
    }

    fn user_version(conn: &Connection) -> i32 {
        conn.pragma_query_value(None, "user_version", |row| row.get(0))
            .expect("user_version")
    }

    fn assert_fork_columns_present(conn: &Connection) {
        let columns = existing_columns(conn).expect("columns");
        for (name, _) in FORK_COLUMNS {
            assert!(columns.iter().any(|c| c == name), "missing {name}");
        }
    }

    #[test]
    fn match_process_is_stored_for_retry() {
        let conn = setup_conn();
        conn.execute(
            "INSERT INTO transcription_history (
                file_name, timestamp, saved, title, transcription_text, context_process
            ) VALUES ('1.wav', 1, 0, 't', 'hi', 'Slack.exe'), ('2.wav', 2, 0, 't', 'hi', NULL)",
            [],
        )
        .expect("insert");
        assert_eq!(
            HistoryManager::find_match_process_with_conn(&conn, 1).unwrap(),
            Some("Slack.exe".to_string())
        );
        assert_eq!(
            HistoryManager::find_match_process_with_conn(&conn, 2).unwrap(),
            None
        );
        assert_eq!(
            HistoryManager::find_match_process_with_conn(&conn, 9).unwrap(),
            None
        );
    }

    #[test]
    fn migration_list_matches_upstream() {
        // Upstream v0.9.8 has four migrations; see MIGRATIONS.
        assert_eq!(MIGRATIONS.len(), 4);
    }

    #[test]
    fn fresh_db_gets_upstream_version_and_fork_columns() {
        let mut conn = Connection::open_in_memory().expect("open in-memory db");
        migrate_schema(&mut conn).expect("migrate");
        assert_eq!(user_version(&conn), 4);
        assert_fork_columns_present(&conn);
        insert_entry(&conn, 100, "new entry", None);
        // Idempotent on the next start.
        migrate_schema(&mut conn).expect("migrate again");
        assert_eq!(user_version(&conn), 4);
    }

    #[test]
    fn upstream_db_gains_fork_columns_and_keeps_entries() {
        let mut conn = Connection::open_in_memory().expect("open in-memory db");
        // An upstream v0.9.8 database.
        Migrations::new(MIGRATIONS.to_vec())
            .to_latest(&mut conn)
            .expect("upstream schema");
        conn.execute(
            "INSERT INTO transcription_history
                (file_name, timestamp, saved, title, transcription_text)
             VALUES ('a.wav', 100, 0, 't', 'old entry')",
            [],
        )
        .expect("insert");

        migrate_schema(&mut conn).expect("migrate");
        assert_eq!(user_version(&conn), 4);
        assert_fork_columns_present(&conn);
        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert_eq!(entry.transcription_text, "old entry");
        assert_eq!(entry.context, None);
        assert_eq!(entry.cleanup_state, None);
    }

    #[test]
    fn older_fork_db_at_version_3_is_upgraded() {
        let mut conn = Connection::open_in_memory().expect("open in-memory db");
        Migrations::new(MIGRATIONS[..3].to_vec())
            .to_latest(&mut conn)
            .expect("old fork schema");
        migrate_schema(&mut conn).expect("migrate");
        assert_eq!(user_version(&conn), 4);
        assert_fork_columns_present(&conn);
    }

    #[test]
    fn db_with_fork_migrations_is_reset_to_upstream_version() {
        for version in [5, 6] {
            let mut conn = setup_conn();
            conn.pragma_update(None, "user_version", version)
                .expect("set version");
            insert_entry(&conn, 100, "kept", None);

            migrate_schema(&mut conn).expect("migrate");
            assert_eq!(user_version(&conn), 4);
            assert_fork_columns_present(&conn);
            // Upstream's migrator now accepts it as up to date.
            Migrations::new(MIGRATIONS.to_vec())
                .to_latest(&mut conn)
                .expect("upstream opens it");
            let entry = HistoryManager::get_latest_entry_with_conn(&conn)
                .expect("fetch")
                .expect("entry");
            assert_eq!(entry.transcription_text, "kept");
        }
    }

    #[test]
    fn version_6_db_from_before_context_process_is_reset_and_completed() {
        let mut conn = setup_conn();
        conn.execute_batch("ALTER TABLE transcription_history DROP COLUMN context_process;")
            .expect("drop column");
        conn.pragma_update(None, "user_version", 6)
            .expect("set version");
        migrate_schema(&mut conn).expect("migrate");
        assert_eq!(user_version(&conn), 4);
        assert_fork_columns_present(&conn);
    }

    #[test]
    fn newer_upstream_version_is_left_alone() {
        let mut conn = setup_conn();
        conn.pragma_update(None, "user_version", 9)
            .expect("set version");
        assert!(migrate_schema(&mut conn).is_err());
        assert_eq!(user_version(&conn), 9);
    }

    #[test]
    fn latest_entries_break_same_second_ties_by_id() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "older", None);
        insert_entry(&conn, 100, "newer", None);

        let latest = HistoryManager::get_latest_completed_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert_eq!(latest.transcription_text, "newer");
        let latest = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert_eq!(latest.transcription_text, "newer");
    }

    #[test]
    fn search_is_unicode_case_insensitive() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "Café ÉCLAIR", None);
        insert_entry(&conn, 200, "Привет, мир", None);
        insert_entry(&conn, 300, "GROẞE Straße", None);
        insert_entry(&conn, 400, "plain ascii", Some("ÜBER polished"));

        let texts = |q: &str| {
            let page = HistoryManager::query_history_page(&conn, None, None, Some(q)).unwrap();
            page.entries
                .into_iter()
                .map(|e| e.transcription_text)
                .collect::<Vec<_>>()
        };
        assert_eq!(texts("éclair"), vec!["Café ÉCLAIR"]);
        assert_eq!(texts("CAFÉ"), vec!["Café ÉCLAIR"]);
        assert_eq!(texts("привет"), vec!["Привет, мир"]);
        assert_eq!(texts("МИР"), vec!["Привет, мир"]);
        // Capital sharp s lowercases to ß; lowercasing never expands to "ss".
        assert_eq!(texts("große"), vec!["GROẞE Straße"]);
        assert_eq!(texts("STRAẞE"), vec!["GROẞE Straße"]);
        assert!(texts("strasse").is_empty());
        assert_eq!(texts("über"), vec!["plain ascii"]);
    }

    #[test]
    fn search_paging_cursor_is_stable_with_same_second_entries() {
        let conn = setup_conn();
        for i in 1..=5 {
            insert_entry(&conn, 100, &format!("tie {i}"), None);
        }
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page =
                HistoryManager::query_history_page(&conn, cursor, Some(2), Some("TIE")).unwrap();
            seen.extend(page.entries.iter().map(|e| e.transcription_text.clone()));
            cursor = page.entries.last().map(|e| e.id);
            if !page.has_more {
                break;
            }
        }
        assert_eq!(seen, vec!["tie 5", "tie 4", "tie 3", "tie 2", "tie 1"]);
    }

    #[test]
    fn retry_keeps_stored_context_when_nothing_new_was_shared() {
        let conn = setup_conn();
        conn.execute(
            "INSERT INTO transcription_history (
                file_name, timestamp, saved, title, transcription_text,
                post_process_requested, context_app, context_title, context_screenshot
            ) VALUES ('1.wav', 1, 0, 't', 'um hi', 1, 'slack', '#general', 1)",
            [],
        )
        .expect("insert");

        // Cleanup skipped on retry: no request, keep the context.
        HistoryManager::update_transcription_with_conn(
            &conn,
            1,
            "hi".into(),
            None,
            None,
            false,
            None,
        )
        .expect("update");
        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert!(!entry.post_process_requested);
        assert_eq!(
            entry.context,
            Some(HistoryContext {
                app: Some("slack".into()),
                title: Some("#general".into()),
                screenshot: true,
            })
        );

        // A request went out without app info: that is what was shared now.
        HistoryManager::update_transcription_with_conn(
            &conn,
            1,
            "hi".into(),
            Some("Hi.".into()),
            None,
            true,
            Some(HistoryContext::default()),
        )
        .expect("update");
        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert!(entry.post_process_requested);
        assert_eq!(entry.context, None);
    }

    #[test]
    fn context_round_trips_and_feeds_recent_apps() {
        let conn = setup_conn();
        let insert = |timestamp: i64, app: Option<&str>, title: Option<&str>, shot: bool| {
            conn.execute(
                "INSERT INTO transcription_history (
                    file_name, timestamp, saved, title, transcription_text,
                    context_app, context_title, context_screenshot
                ) VALUES (?1, ?2, 0, 't', 'text', ?3, ?4, ?5)",
                params![format!("{timestamp}.wav"), timestamp, app, title, shot],
            )
            .expect("insert");
        };
        insert(1, Some("Slack"), Some("#general"), true);
        insert(2, Some("Code"), None, false);
        insert(3, None, None, false);
        insert(4, Some("Slack"), None, false);

        let latest = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch")
            .expect("entry");
        assert_eq!(
            latest.context,
            Some(HistoryContext {
                app: Some("Slack".into()),
                title: None,
                screenshot: false,
            })
        );
        let first = HistoryManager::query_history_page(&conn, None, None, None)
            .expect("page")
            .entries
            .pop()
            .expect("oldest");
        assert_eq!(first.context.unwrap().title.as_deref(), Some("#general"));

        let apps = HistoryManager::get_recent_context_apps_with_conn(&conn, 10).expect("apps");
        assert_eq!(apps, vec!["Slack".to_string(), "Code".to_string()]);
        let one = HistoryManager::get_recent_context_apps_with_conn(&conn, 1).expect("apps");
        assert_eq!(one, vec!["Slack".to_string()]);
    }

    #[test]
    fn empty_history_context_is_none() {
        assert_eq!(HistoryContext::default().into_option(), None);
        let shot_only = HistoryContext {
            screenshot: true,
            ..Default::default()
        };
        assert_eq!(shot_only.clone().into_option(), Some(shot_only));
    }
}
