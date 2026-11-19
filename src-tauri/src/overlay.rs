//! Overlay windows: one per monitor, created on demand and torn down afterwards.
//!
//! Each window covers exactly one monitor (`position` = logical origin,
//! `inner_size` = logical size) and loads the SPA at `index.html?view=overlay`.
//!
//! Two macOS quirks shape this module:
//! * A transparent, borderless window created with `visible(false)` and later
//!   revealed with `show()` stays off-screen (`kCGWindowIsOnscreen` stays -1),
//!   so windows are created visible and *closed* rather than hidden.
//! * Creating the window before capturing means the overlay can appear in its
//!   own screenshot, so capture happens first and windows are built after.

use anyhow::Result;
use std::time::Duration;
use tauri::{AppHandle, Manager, WebviewWindow};

/// The tool the overlay should auto-start once the user frames a selection.
/// Set by the tray's 长截图 entry before `commands::start_capture`, consumed
/// by it when the overlay windows are built.
static BOOT_TOOL: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn request_boot_tool(tool: &str) {
    *BOOT_TOOL.lock().unwrap() = Some(tool.to_string());
}

/// Read-and-clear: one boot request must never leak into a later session.
pub fn take_boot_tool() -> String {
    BOOT_TOOL.lock().unwrap().take().unwrap_or_default()
}

/// URL for an overlay window; `CAP_PHOTO_DEBUG_BOX` preselects a region so the
/// toolbar can be driven without a mouse during automation.
fn overlay_url(monitor_id: u32, boot: &str) -> String {
    let mut url = format!("index.html?view=overlay&monitor={monitor_id}");
    if !boot.is_empty() {
        url.push_str(&format!("&boot={boot}"));
    }
    if std::env::var("CAP_PHOTO_DEBUG_BOX").is_ok() {
        url.push_str("&debug=box");
    }
    url
}

pub fn label_for(monitor_id: u32) -> String {
    format!("overlay-{monitor_id}")
}

/// Create the overlay window for one monitor (replacing any stale one).
pub fn create(
    app: &AppHandle,
    monitor_id: u32,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    boot: &str,
) -> Result<WebviewWindow> {
    let label = label_for(monitor_id);
    if let Some(old) = app.get_webview_window(&label) {
        let _ = old.close();
    }

    log::info!("creating overlay window {label} at {x},{y} {width}x{height}");
    let win = crate::windows::layer_window(
        app,
        &label,
        &overlay_url(monitor_id, boot),
        "Shutter",
        x as f64,
        y as f64,
        width as f64,
        height as f64,
        true,
    )?;

    log::info!("overlay window {label} created");
    Ok(win)
}

/// Bring an overlay to the front and confirm it landed on screen.
///
/// `monitor` is the display the window was built for: its logical origin is the
/// only trustworthy frame reference available here.
pub fn reveal(win: &WebviewWindow, monitor: &crate::types::MonitorInfo) -> Result<()> {
    let label = win.label().to_string();
    let (mx, my) = (f64::from(monitor.x), f64::from(monitor.y));
    win.set_always_on_top(true)
        .map_err(|e| anyhow::anyhow!("always-on-top {label}: {e}"))?;
    // set_always_on_top resets the level to floating (below the Dock);
    // re-raise so the toolbar is never covered by the Dock or menu bar.
    crate::windows::raise_above_dock(win);
    // focus_when_ready below keeps calling show/set_focus while the webview
    // boots; if any of that re-asserts the floating level, re-raise once the
    // dust has settled.
    {
        let later = win.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(800));
            crate::windows::raise_above_dock(&later);
        });
    }
    let _ = win.set_focus();

    // A transparent borderless window can be attached to the window server with
    // the "not on screen" bit still set. Toggling alpha and ordering front is
    // what actually makes macOS composite it.
    // The overlay is a borderless opaque-to-inputs window in an Accessory app:
    // without an explicit activation the first click/keystroke only activates
    // the window and is swallowed, which reads as "the toolbar does nothing".
    crate::windows::focus_when_ready(win.clone());

    #[cfg(target_os = "macos")]
    {
        use tauri::Manager;
        if let Some(_) = win.app_handle().get_webview_window(&label) {
            win.set_ignore_cursor_events(false).ok();
            win.show().ok();
            // Nudge the frame: some macOS versions ignore the first show of a
            // window created with a non-standard level.
            //
            // The nudge is written in logical points, so its reference is the
            // monitor's logical origin rather than a readback of the live frame:
            // `outer_position()` reports physical pixels while
            // `set_position(LogicalPosition)` is consumed as Cocoa points, so on
            // a scaled display whose origin is not the desktop origin, a
            // readback-then-write lands the window at double that origin — off
            // the desktop entirely.
            let _ = win.set_position(tauri::LogicalPosition::new(mx, my + 1.0));
            let _ = win.set_position(tauri::LogicalPosition::new(mx, my));
        }
    }
    // Expected values are the monitor's logical frame; the measured pair is in
    // physical pixels. They differ by exactly `scale_factor` unless the window
    // server has clamped or re-parented the window.
    let size = win.inner_size().map(|s| (s.width, s.height)).unwrap_or((0, 0));
    let pos = win.outer_position().map(|p| (p.x, p.y)).unwrap_or((0, 0));
    log::info!(
        "overlay {label} revealed: visible={:?} expected_logical_origin=({mx}, {my}) \
         expected_logical_size=({}x{}) actual_physical_pos={pos:?} actual_physical_size={size:?} \
         scale_factor={}",
        win.is_visible(),
        monitor.width,
        monitor.height,
        win.scale_factor().unwrap_or(0.0),
    );
    Ok(())
}

/// Destroy every overlay window. Closing is deliberate: `hide()` + `show()`
/// leaves a transparent macOS window off-screen on the next reveal.
pub fn destroy_all(app: &AppHandle) {
    for (label, win) in app.webview_windows() {
        if label.starts_with("overlay-") {
            if let Err(e) = win.close() {
                log::warn!("close {label}: {e}");
            }
        }
    }
}

/// Hand key-window status to the overlay the pointer is on.
///
/// macOS delivers `mouseMoved` to the key window only, so an overlay that is
/// not key sees no pointer movement at all — window snapping, the loupe and
/// the eyedropper then track the cursor on one display only. A click still
/// works anywhere, because the mouse-down makes its own window key first. With
/// one overlay per display, exactly one display can ever be key, so the key
/// window has to follow the pointer.
///
/// The pointer is read from the hardware event source (the same call the
/// auto-scroller uses) and only the display *change* triggers a focus call, so
/// movement inside a display — including a selection drag — leaves the window
/// stack alone.
#[cfg(target_os = "macos")]
pub fn follow_pointer(app: &AppHandle, monitors: Vec<crate::types::MonitorInfo>) {
    let app = app.clone();
    std::thread::spawn(move || {
        let mut over: Option<u32> = None;
        // The overlays are the session: once confirm/cancel has closed them
        // there is nothing left to track. The bound is a safety net for a
        // session that ends without the windows ever being observed closed.
        for _ in 0..10_000 {
            let windows = app.webview_windows();
            if !windows.keys().any(|label| label.starts_with("overlay-")) {
                break;
            }
            let (px, py) = crate::scroll::mouse_location();
            if px.is_finite() && py.is_finite() {
                let hit = monitors.iter().find(|m| {
                    px >= f64::from(m.x)
                        && px < f64::from(m.x) + f64::from(m.width)
                        && py >= f64::from(m.y)
                        && py < f64::from(m.y) + f64::from(m.height)
                });
                let hit = hit.map(|m| m.id);
                if hit != over {
                    over = hit;
                    if let Some(id) = hit {
                        let label = label_for(id);
                        if let Some(win) = windows.get(&label) {
                            log::info!("pointer moved to monitor {id}: focusing {label}");
                            let _ = win.set_focus();
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(60));
        }
    });
}

/// No other platform restricts pointer-move delivery to the key window.
#[cfg(not(target_os = "macos"))]
pub fn follow_pointer(_app: &AppHandle, _monitors: Vec<crate::types::MonitorInfo>) {}
