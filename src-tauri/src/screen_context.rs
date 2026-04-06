use base64::Engine;
use image::ImageFormat;
use std::io::Cursor;
use xcap::Window;

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
    let windows = Window::all().ok()?;
    let focused = windows
        .into_iter()
        .find(|w| w.is_focused().unwrap_or(false))?;

    let app_name = focused.app_name().unwrap_or_default();
    let window_title = focused.title().unwrap_or_default();

    let capture = focused.capture_image().ok()?;

    let mut png_bytes = Cursor::new(Vec::new());
    capture
        .write_to(&mut png_bytes, ImageFormat::Png)
        .ok()?;

    let image_base64 = base64::engine::general_purpose::STANDARD.encode(png_bytes.into_inner());

    Some(ScreenContext {
        image_base64,
        window_title,
        app_name,
    })
}
