use base64::Engine;
use image::ImageFormat;
use log::{debug, warn};
use std::io::Cursor;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use xcap::Window;

/// How long post-processing waits for an in-flight capture before giving up
/// and falling back to text-only post-processing. Capture normally finishes
/// long before transcription does, so this only matters if the OS capture API
/// hangs.
const CAPTURE_WAIT_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone)]
pub struct ScreenContext {
    /// Base64-encoded PNG screenshot
    pub image_base64: String,
    /// Window title of the focused window
    pub window_title: String,
    /// Application/process name
    pub app_name: String,
}

/// Captures a screenshot of the currently focused window.
/// Returns `None` if no focused window is found or capture fails.
pub fn capture_focused_window() -> Option<ScreenContext> {
    let windows = match Window::all() {
        Ok(windows) => windows,
        Err(e) => {
            warn!("Screen context: failed to enumerate windows: {}", e);
            return None;
        }
    };
    let Some(focused) = windows
        .into_iter()
        .find(|w| w.is_focused().unwrap_or(false))
    else {
        debug!("Screen context: no focused window found");
        return None;
    };

    let app_name = focused.app_name().unwrap_or_default();
    let window_title = focused.title().unwrap_or_default();

    let capture = match focused.capture_image() {
        Ok(capture) => capture,
        Err(e) => {
            warn!("Screen context: failed to capture focused window: {}", e);
            return None;
        }
    };

    let mut png_bytes = Cursor::new(Vec::new());
    if let Err(e) = capture.write_to(&mut png_bytes, ImageFormat::Png) {
        warn!("Screen context: failed to encode screenshot: {}", e);
        return None;
    }

    let image_base64 = base64::engine::general_purpose::STANDARD.encode(png_bytes.into_inner());

    Some(ScreenContext {
        image_base64,
        window_title,
        app_name,
    })
}

/// A screen capture that was started when recording began and may still be in
/// flight. Resolve it only when the vision post-processing path needs it.
pub struct PendingScreenContext(oneshot::Receiver<Option<ScreenContext>>);

impl PendingScreenContext {
    /// Wait (bounded) for the capture to finish.
    pub async fn resolve(self) -> Option<ScreenContext> {
        match tokio::time::timeout(CAPTURE_WAIT_TIMEOUT, self.0).await {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(_)) => {
                warn!("Screen context: capture thread ended without a result");
                None
            }
            Err(_) => {
                warn!(
                    "Screen context: capture did not finish within {:?}",
                    CAPTURE_WAIT_TIMEOUT
                );
                None
            }
        }
    }
}

/// Managed state that hands the screenshot taken at hotkey press (start) to
/// the transcription pipeline (stop).
#[derive(Default)]
pub struct ScreenContextSlot {
    pending: Mutex<Option<PendingScreenContext>>,
}

impl ScreenContextSlot {
    fn lock(&self) -> MutexGuard<'_, Option<PendingScreenContext>> {
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Capture the focused window on a background thread so the screenshot
    /// (and its PNG encoding) never adds latency to starting the recording.
    /// Replaces any capture left over from a previous recording.
    pub fn begin_capture(&self) {
        let (tx, rx) = oneshot::channel();
        *self.lock() = Some(PendingScreenContext(rx));

        let spawned = std::thread::Builder::new()
            .name("screen-context-capture".to_string())
            .spawn(move || {
                let started = Instant::now();
                let ctx = capture_focused_window();
                if let Some(ctx) = &ctx {
                    debug!(
                        "Screen context captured from '{}' (title: {} chars, {} bytes base64) in {:?}",
                        ctx.app_name,
                        ctx.window_title.chars().count(),
                        ctx.image_base64.len(),
                        started.elapsed()
                    );
                }
                let _ = tx.send(ctx);
            });

        if let Err(e) = spawned {
            warn!("Screen context: failed to spawn capture thread: {}", e);
            self.clear();
        }
    }

    /// Drop any pending or captured screenshot.
    pub fn clear(&self) {
        self.lock().take();
    }

    /// Take the capture for the recording that just stopped.
    pub fn take(&self) -> Option<PendingScreenContext> {
        self.lock().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_context() -> ScreenContext {
        ScreenContext {
            image_base64: "aGVsbG8=".to_string(),
            window_title: "Inbox".to_string(),
            app_name: "mail".to_string(),
        }
    }

    #[test]
    fn take_returns_pending_once() {
        let slot = ScreenContextSlot::default();
        let (tx, rx) = oneshot::channel();
        *slot.lock() = Some(PendingScreenContext(rx));
        tx.send(Some(sample_context())).unwrap();

        let pending = slot.take().expect("pending capture");
        assert!(slot.take().is_none());

        let ctx = tauri::async_runtime::block_on(pending.resolve()).expect("context");
        assert_eq!(ctx.app_name, "mail");
    }

    #[test]
    fn clear_drops_pending_capture() {
        let slot = ScreenContextSlot::default();
        let (_tx, rx) = oneshot::channel();
        *slot.lock() = Some(PendingScreenContext(rx));
        slot.clear();
        assert!(slot.take().is_none());
    }

    #[test]
    fn resolve_returns_none_when_sender_dropped() {
        let (tx, rx) = oneshot::channel::<Option<ScreenContext>>();
        drop(tx);
        let ctx = tauri::async_runtime::block_on(PendingScreenContext(rx).resolve());
        assert!(ctx.is_none());
    }
}
