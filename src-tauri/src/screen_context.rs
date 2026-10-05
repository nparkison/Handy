//! Press-time context capture.
//!
//! When a dictation starts, a background thread notes the focused app (see
//! [`crate::app_context`]) and, only when a matching app rule opted in and the
//! provider can read images, screenshots the active window. Nothing here adds
//! latency to starting the recording, and screenshots stay in memory: they are
//! never written to disk or History.

#[cfg(windows)]
use crate::app_context::capture_app_info;
use crate::app_context::{match_rule, screenshot_wanted, AppInfo};
use crate::settings::AppSettings;
use base64::Engine;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::RgbaImage;
use log::{debug, warn};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// The app info is ready within milliseconds of the press; this only matters
/// if the OS call hangs.
const APP_INFO_WAIT_TIMEOUT: Duration = Duration::from_millis(500);

/// How long cleanup waits for an in-flight screenshot before going on without
/// it. The wait happens before the request is sent, so it never counts against
/// the cleanup time limit.
const CAPTURE_WAIT_TIMEOUT: Duration = Duration::from_secs(3);

/// Screenshots are downscaled so their longest edge is at most this.
pub const SCREENSHOT_MAX_EDGE: u32 = 1024;
const JPEG_QUALITY: u8 = 75;

/// An in-memory, JPEG-encoded screenshot of the active window.
#[derive(Clone)]
pub struct Screenshot {
    pub base64: String,
}

impl Screenshot {
    pub const MIME: &'static str = "image/jpeg";
}

impl std::fmt::Debug for Screenshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Screenshot({} bytes base64)", self.base64.len())
    }
}

/// Size that fits `width` x `height` within `max_edge` on its longest side,
/// keeping the aspect ratio. Images already small enough are left alone.
pub fn downscaled_size(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max_edge || longest == 0 {
        return (width, height);
    }
    let scale = |side: u32| {
        let scaled =
            (u64::from(side) * u64::from(max_edge) + u64::from(longest) / 2) / u64::from(longest);
        (scaled as u32).max(1)
    };
    (scale(width), scale(height))
}

/// Downscale and JPEG-encode a capture.
fn encode_screenshot(capture: RgbaImage) -> Option<Screenshot> {
    let (width, height) = downscaled_size(capture.width(), capture.height(), SCREENSHOT_MAX_EDGE);
    let image = if (width, height) == capture.dimensions() {
        capture
    } else {
        image::imageops::resize(&capture, width, height, FilterType::Triangle)
    };
    // JPEG has no alpha channel.
    let rgb = image::DynamicImage::ImageRgba8(image).into_rgb8();
    let mut jpeg = Vec::new();
    if let Err(e) = JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY).encode_image(&rgb) {
        warn!("Screen context: failed to encode screenshot: {}", e);
        return None;
    }
    Some(Screenshot {
        base64: base64::engine::general_purpose::STANDARD.encode(jpeg),
    })
}

/// Largest window side captured (before downscaling); bigger windows are
/// skipped rather than decoding a huge bitmap.
const MAX_CAPTURE_EDGE: u32 = 8_192;

/// The focused window, found by enumerating all windows (macOS/Linux).
#[cfg(not(windows))]
pub fn focused_window() -> Option<xcap::Window> {
    match xcap::Window::all() {
        Ok(windows) => windows
            .into_iter()
            .find(|w| w.is_focused().unwrap_or(false)),
        Err(e) => {
            warn!("Screen context: failed to enumerate windows: {}", e);
            None
        }
    }
}

/// The xcap window with this id (on Windows, the HWND the rule matched).
#[cfg(windows)]
fn window_by_id(id: isize) -> Option<xcap::Window> {
    let target = u32::try_from(id).ok()?;
    match xcap::Window::all() {
        Ok(windows) => windows.into_iter().find(|w| w.id().ok() == Some(target)),
        Err(e) => {
            warn!("Screen context: failed to enumerate windows: {}", e);
            None
        }
    }
}

/// Screenshot of one window only (never the full screen).
fn capture_window(window: &xcap::Window) -> Option<Screenshot> {
    if window.is_minimized().unwrap_or(false) {
        debug!("Screen context: window is minimized; no screenshot");
        return None;
    }
    let (width, height) = (window.width().unwrap_or(0), window.height().unwrap_or(0));
    if width == 0 || height == 0 || width > MAX_CAPTURE_EDGE || height > MAX_CAPTURE_EDGE {
        debug!("Screen context: window size {width}x{height} not captured");
        return None;
    }
    match window.capture_image() {
        Ok(capture) => encode_screenshot(capture),
        Err(e) => {
            warn!("Screen context: failed to capture focused window: {}", e);
            None
        }
    }
}

/// Context captured at press for the recording that is about to stop.
pub struct PendingPressContext {
    app: oneshot::Receiver<Option<AppInfo>>,
    screenshot: oneshot::Receiver<Option<Screenshot>>,
}

impl PendingPressContext {
    /// Wait (bounded) for the app info. The screenshot, if one is coming,
    /// stays pending in the returned [`PendingScreenshot`].
    pub async fn resolve_app(self) -> (Option<AppInfo>, PendingScreenshot) {
        let app = match tokio::time::timeout(APP_INFO_WAIT_TIMEOUT, self.app).await {
            Ok(Ok(info)) => info,
            Ok(Err(_)) => None,
            Err(_) => {
                warn!("App context: lookup did not finish within {APP_INFO_WAIT_TIMEOUT:?}");
                None
            }
        };
        (app, PendingScreenshot(self.screenshot))
    }
}

/// A screenshot that may still be in flight.
pub struct PendingScreenshot(oneshot::Receiver<Option<Screenshot>>);

impl PendingScreenshot {
    /// Wait (bounded) for the screenshot. Resolves to `None` right away when
    /// no screenshot was taken.
    pub async fn resolve(self) -> Option<Screenshot> {
        match tokio::time::timeout(CAPTURE_WAIT_TIMEOUT, self.0).await {
            Ok(Ok(shot)) => shot,
            Ok(Err(_)) => None,
            Err(_) => {
                warn!("Screen context: capture did not finish within {CAPTURE_WAIT_TIMEOUT:?}");
                None
            }
        }
    }
}

/// Managed state that hands the context captured at hotkey press (start) to
/// the transcription pipeline (stop).
#[derive(Default)]
pub struct PressContextSlot {
    pending: Mutex<Option<PendingPressContext>>,
}

impl PressContextSlot {
    fn lock(&self) -> MutexGuard<'_, Option<PendingPressContext>> {
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Note the focused app on a background thread and, when a matching app
    /// rule asks for it (and cleanup will run with a vision provider),
    /// screenshot the active window. Replaces any leftover capture.
    pub fn begin_capture(&self, settings: &AppSettings, cleanup_will_run: bool) {
        let (app_tx, app_rx) = oneshot::channel();
        let (shot_tx, shot_rx) = oneshot::channel();
        *self.lock() = Some(PendingPressContext {
            app: app_rx,
            screenshot: shot_rx,
        });

        // Only clone the settings when some rule could ask for a screenshot.
        let screenshot_settings = (cleanup_will_run
            && settings
                .app_rules
                .iter()
                .any(|rule| screenshot_wanted(settings, Some(rule))))
        .then(|| settings.clone());
        let spawned = std::thread::Builder::new()
            .name("press-context-capture".to_string())
            .spawn(move || {
                let started = Instant::now();
                // One window lookup per press: the screenshot uses the same
                // window the rule matched, never whatever is focused later.
                #[cfg(windows)]
                let info = capture_app_info();
                #[cfg(not(windows))]
                let focused = focused_window();
                #[cfg(not(windows))]
                let info = focused
                    .as_ref()
                    .and_then(crate::app_context::app_info_from_window);
                let wants_shot = screenshot_settings.as_ref().is_some_and(|settings| {
                    screenshot_wanted(settings, match_rule(&settings.app_rules, info.as_ref()))
                });
                debug!(
                    "App context captured in {:?} (app: {:?}, title: {} chars)",
                    started.elapsed(),
                    info.as_ref().map(|i| i.app_name.as_str()),
                    info.as_ref()
                        .and_then(|i| i.window_title.as_ref())
                        .map_or(0, |t| t.chars().count())
                );
                #[cfg(windows)]
                let window_id = info.as_ref().and_then(|i| i.window);
                let _ = app_tx.send(info);
                if wants_shot {
                    #[cfg(windows)]
                    let window = window_id.and_then(window_by_id);
                    #[cfg(not(windows))]
                    let window = focused;
                    let shot = match window {
                        Some(window) => capture_window(&window),
                        None => {
                            debug!("Screen context: matched window not found; no screenshot");
                            None
                        }
                    };
                    debug!("Screen context: {:?} in {:?}", shot, started.elapsed());
                    let _ = shot_tx.send(shot);
                }
                // Otherwise dropping shot_tx resolves the screenshot to None.
            });

        if let Err(e) = spawned {
            warn!("Press context: failed to spawn capture thread: {}", e);
            self.clear();
        }
    }

    /// Drop any pending or captured context.
    pub fn clear(&self) {
        self.lock().take();
    }

    /// Take the capture for the recording that just stopped.
    pub fn take(&self) -> Option<PendingPressContext> {
        self.lock().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_keeps_aspect_ratio_within_max_edge() {
        assert_eq!(downscaled_size(2560, 1440, 1024), (1024, 576));
        assert_eq!(downscaled_size(1440, 2560, 1024), (576, 1024));
        assert_eq!(downscaled_size(3000, 3000, 1024), (1024, 1024));
        assert_eq!(downscaled_size(1366, 768, 1024), (1024, 576));
    }

    #[test]
    fn downscale_leaves_small_images_alone() {
        assert_eq!(downscaled_size(800, 600, 1024), (800, 600));
        assert_eq!(downscaled_size(1024, 10, 1024), (1024, 10));
        assert_eq!(downscaled_size(0, 0, 1024), (0, 0));
    }

    #[test]
    fn downscale_never_collapses_a_side_to_zero() {
        assert_eq!(downscaled_size(10_000, 1, 1024), (1024, 1));
    }

    #[test]
    fn encoded_screenshot_is_a_downscaled_jpeg() {
        let capture = RgbaImage::from_pixel(2048, 1024, image::Rgba([10, 20, 30, 255]));
        let shot = encode_screenshot(capture).expect("encodes");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(shot.base64)
            .unwrap();
        assert_eq!(&bytes[..3], &[0xFF, 0xD8, 0xFF]); // JPEG SOI marker
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (1024, 512));
    }

    #[test]
    fn take_returns_pending_once_and_clear_drops_it() {
        let slot = PressContextSlot::default();
        let (app_tx, app_rx) = oneshot::channel();
        let (_shot_tx, shot_rx) = oneshot::channel();
        *slot.lock() = Some(PendingPressContext {
            app: app_rx,
            screenshot: shot_rx,
        });
        app_tx
            .send(Some(AppInfo::from_process_path("mail.exe", None)))
            .unwrap();

        let pending = slot.take().expect("pending capture");
        assert!(slot.take().is_none());
        let (info, _) = tauri::async_runtime::block_on(pending.resolve_app());
        assert_eq!(info.unwrap().app_name, "mail");

        let (_a, app_rx) = oneshot::channel();
        let (_s, shot_rx) = oneshot::channel();
        *slot.lock() = Some(PendingPressContext {
            app: app_rx,
            screenshot: shot_rx,
        });
        slot.clear();
        assert!(slot.take().is_none());
    }

    #[test]
    fn dropped_screenshot_sender_resolves_to_none() {
        let (tx, rx) = oneshot::channel::<Option<Screenshot>>();
        drop(tx);
        let shot = tauri::async_runtime::block_on(PendingScreenshot(rx).resolve());
        assert!(shot.is_none());
    }
}
