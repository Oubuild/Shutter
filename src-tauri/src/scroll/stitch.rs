//! Incremental stitching of accepted frames into the accumulated long image.
//!
//! Mirrors the reference `image_stitcher` + `compute_top`: frames are appended
//! onto a growing buffer at the estimated (and verifier-gated) offset, with
//! sticky-region removal on BOTH ends (reference `margins::detect_fixed_margins`
//! + `stitch_incremental_with_fixed_margins`):
//! * a confirmed fixed header is cropped at finish;
//! * a confirmed fixed footer is trimmed from every append and stripped out
//!   of the base frame once detected;
//! * the last few appended frames are kept as a ring so the pipeline can
//!   rescue a frame against an older baseline (reference `frame_ring`).

use crate::scroll::pipeline::FrameDecision;
use image::{imageops, RgbaImage};
use std::collections::VecDeque;

/// A run of rows that is byte-identical between consecutive frames near the
/// same y position across ≥ this many frames is treated as a fixed bar.
const STICKY_MIN_FRAMES: u32 = 3;

/// Max height of the header/footer band considered for sticky detection.
const STICKY_MAX_BAND: u32 = 220;

/// A leading/trailing equal run shorter than this is not treated as a bar at
/// all (rounded corners, hairlines, banding noise).
const STICKY_MIN_ROWS: u32 = 24;

/// How many appended frames the rescue ring keeps (reference `RING`, read
/// through `nth_back(1/2)` → at least 3).
const RING_LEN: usize = 3;

/// Consecutive accepted frames required before a column is treated as fixed.
const FIXED_COLUMN_STREAK: u8 = 3;

/// Sample interval used to compare columns without scanning every pixel.
const FIXED_COLUMN_SAMPLE_STEP: u32 = 4;

/// Minimum horizontal span treated as one fixed region.
const FIXED_REGION_MIN_WIDTH: usize = 16;

/// Maximum fraction of changed samples for a column to count as fixed.
const FIXED_COLUMN_MAX_CHANGE: f64 = 0.01;

/// The growing stitched image plus everything needed to append safely.
pub struct Stitcher {
    /// The accumulated image (width = selection width, grows downward).
    image: RgbaImage,
    /// The last frame as appended (post dedup), for the next alignment.
    last_frame: RgbaImage,
    /// Offsets actually committed (for seam bias + direction checks).
    history: Vec<i64>,
    /// Consecutive frames' leading-equal-row depths, for sticky detection.
    candidate_depths: Vec<u32>,
    /// Rows at the top confirmed to be a fixed (sticky) region.
    sticky_top: u32,
    /// Consecutive frames' trailing-equal-row depths (bottom bar detection).
    bottom_candidate_depths: Vec<u32>,
    /// Rows at the bottom confirmed to be a fixed (sticky) region.
    sticky_bottom: u32,
    /// This frame's trailing-equal depth (≥ STICKY_MIN_ROWS or 0): the
    /// provisional footer trimmed from the append before confirmation.
    last_bottom_depth: u32,
    /// Height of the very first frame; the footer band it carries sits at
    /// rows [base_height − sticky_bottom, base_height) until stripped.
    base_height: u32,
    /// Whether the base frame's footer band has been removed.
    bottom_stripped: bool,
    /// The last RING_LEN appended frames, oldest first (rescue baselines).
    ring: VecDeque<RgbaImage>,
    /// Diagnostics counter.
    appended: u32,
    /// Consecutive frames in which each x column stayed visually fixed.
    fixed_column_streaks: Vec<u8>,
    /// Columns confirmed as fixed across the viewport.
    fixed_columns: Vec<bool>,
}

fn row_hash(img: &RgbaImage, y: u32) -> u64 {
    let w = img.width() as usize;
    let mut h: u64 = 1469598103934665603;
    for x in 0..w {
        let p = img.get_pixel(x as u32, y);
        for c in 0..3 {
            h ^= p.0[c] as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

impl Stitcher {
    /// First frame becomes the base of the accumulated image.
    pub fn new(first: RgbaImage) -> Self {
        let base_height = first.height();
        let width = first.width() as usize;
        let mut ring = VecDeque::with_capacity(RING_LEN);
        ring.push_back(first.clone());
        Self {
            image: first.clone(),
            last_frame: first,
            history: Vec::new(),
            candidate_depths: Vec::new(),
            sticky_top: 0,
            bottom_candidate_depths: Vec::new(),
            sticky_bottom: 0,
            last_bottom_depth: 0,
            base_height,
            bottom_stripped: false,
            ring,
            appended: 1,
            fixed_column_streaks: vec![0; width],
            fixed_columns: vec![false; width],
        }
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn history(&self) -> &[i64] {
        &self.history
    }

    pub fn appended_frames(&self) -> u32 {
        self.appended
    }

    /// The appended frame `back` generations before the newest one
    /// (0 = last appended, 1 = the one before it); None when the ring is too
    /// short (reference `frame_ring::nth_back`).
    pub fn rescue_baseline(&self, back: usize) -> Option<&RgbaImage> {
        let len = self.ring.len();
        len.checked_sub(1 + back).map(|i| &self.ring[i])
    }

    /// The last committed append offset, in pixels.
    pub fn last_committed(&self) -> Option<i64> {
        self.history.last().copied()
    }

    /// Rows of `img` [y0, y1) as an owned buffer.
    fn rows(img: &RgbaImage, y0: u32, y1: u32) -> RgbaImage {
        imageops::crop_imm(img, 0, y0, img.width(), y1 - y0).to_image()
    }

    /// Detect a fixed top band: rows that are identical at the SAME
    /// frame-local y across `STICKY_MIN_FRAMES` consecutive frames
    /// (reference `detect_fixed_margins` / `MAC_FIXED_TILE_STREAKS`). A
    /// scrolling page never repeats a row at the same y; a fixed header does.
    fn update_sticky_top(&mut self, frame: &RgbaImage) {
        // Depth of the leading run of rows equal to the previous frame's rows
        // at the same y.
        let limit = frame.height().min(STICKY_MAX_BAND);
        let mut depth = 0u32;
        while depth < limit && row_hash(frame, depth) == row_hash(&self.last_frame, depth) {
            depth += 1;
        }
        // Track the streak: a candidate becomes confirmed once it repeats for
        // STICKY_MIN_FRAMES frames (2 in the pair + current).
        if depth >= STICKY_MIN_ROWS {
            self.candidate_depths.push(depth);
        } else {
            self.candidate_depths.clear();
        }
        if self.candidate_depths.len() >= STICKY_MIN_FRAMES as usize {
            let confirmed = *self.candidate_depths.iter().min().unwrap();
            if confirmed > self.sticky_top {
                self.sticky_top = confirmed;
                log::info!("[长截图] 固定栏 header={}px", self.sticky_top);
            }
        }
    }

    /// The mirror of `update_sticky_top` for the frame's bottom: the depth of
    /// the trailing run of rows identical at the same frame-local y. Records
    /// this frame's depth as the provisional footer trim and confirms a
    /// permanent `sticky_bottom` off the streak.
    ///
    /// Trimming provisionally (before streak confirmation) is what keeps a
    /// footer out of the middle of the image from the very first append; the
    /// streak only upgrades it to a confirmed bar that also gets stripped
    /// from the base frame. A coincidental equal band (uniform spacing that
    /// happens to span two viewports) costs at most its own rows on that one
    /// frame and breaks the streak immediately.
    fn update_sticky_bottom(&mut self, frame: &RgbaImage) {
        let h = frame.height();
        let limit = h.min(STICKY_MAX_BAND);
        let mut depth = 0u32;
        while depth < limit
            && row_hash(frame, h - 1 - depth) == row_hash(&self.last_frame, h - 1 - depth)
        {
            depth += 1;
        }
        self.last_bottom_depth = if depth >= STICKY_MIN_ROWS {
            depth
        } else {
            0
        };
        if self.last_bottom_depth > 0 {
            self.bottom_candidate_depths.push(depth);
        } else {
            self.bottom_candidate_depths.clear();
        }
        if self.bottom_candidate_depths.len() >= STICKY_MIN_FRAMES as usize {
            let confirmed = *self.bottom_candidate_depths.iter().min().unwrap();
            if confirmed > self.sticky_bottom {
                self.sticky_bottom = confirmed;
                log::info!("[长截图] 固定栏 footer={}px", self.sticky_bottom);
            }
        }
    }

    /// Remove the base frame's footer band from the accumulated image once a
    /// sticky bottom is confirmed: rows [base_height − sticky_bottom,
    /// base_height) are chrome the appends already bypass. Content appended
    /// above/below the band joins seamlessly because the band sat exactly
    /// between them.
    fn strip_base_footer(&mut self) {
        if self.bottom_stripped || self.sticky_bottom == 0 {
            return;
        }
        let h = self.image.height();
        if h <= self.base_height {
            // Nothing appended below the base yet; finish() crops the tail.
            return;
        }
        let cut_lo = self.base_height.saturating_sub(self.sticky_bottom);
        if cut_lo == 0 {
            return;
        }
        let mut grown = RgbaImage::new(self.image.width(), h - self.sticky_bottom);
        imageops::overlay(&mut grown, &Self::rows(&self.image, 0, cut_lo), 0, 0);
        imageops::overlay(
            &mut grown,
            &Self::rows(&self.image, self.base_height, h),
            0,
            cut_lo as i64,
        );
        self.image = grown;
        self.bottom_stripped = true;
        log::info!("[长截图] 去除基帧固定底栏 {}px", self.sticky_bottom);
    }

    fn update_fixed_columns(&mut self, frame: &RgbaImage) {
        if frame.dimensions() != self.last_frame.dimensions() {
            return;
        }
        let (width, height) = frame.dimensions();
        let samples = height.div_ceil(FIXED_COLUMN_SAMPLE_STEP).max(1);
        for x in 0..width {
            let mut changed = 0u32;
            for y in (0..height).step_by(FIXED_COLUMN_SAMPLE_STEP as usize) {
                let a = self.last_frame.get_pixel(x, y).0;
                let b = frame.get_pixel(x, y).0;
                if (0..3).any(|channel| a[channel].abs_diff(b[channel]) > 3) {
                    changed += 1;
                }
            }
            let index = x as usize;
            if changed as f64 / samples as f64 <= FIXED_COLUMN_MAX_CHANGE {
                self.fixed_column_streaks[index] =
                    self.fixed_column_streaks[index].saturating_add(1);
                if self.fixed_column_streaks[index] >= FIXED_COLUMN_STREAK {
                    self.fixed_columns[index] = true;
                }
            } else {
                self.fixed_column_streaks[index] = 0;
            }
        }
        let mut x = 0;
        while x < self.fixed_columns.len() {
            if !self.fixed_columns[x] {
                x += 1;
                continue;
            }
            let start = x;
            while x < self.fixed_columns.len() && self.fixed_columns[x] {
                x += 1;
            }
            if x - start < FIXED_REGION_MIN_WIDTH {
                self.fixed_columns[start..x].fill(false);
            }
        }
    }

    fn clear_fixed_columns(&mut self, y_start: u32) {
        if !self.fixed_columns.iter().any(|fixed| *fixed) {
            return;
        }
        let y_end = self.image.height();
        for (x, fixed) in self.fixed_columns.iter().enumerate() {
            if *fixed {
                for y in y_start..y_end {
                    self.image.put_pixel(x as u32, y, image::Rgba([0, 0, 0, 0]));
                }
            }
        }
    }

    /// Append a frame at the estimated offset. Returns the rows actually added.
    ///
    /// Geometry (same as the pipeline's): the frame's content above the fixed
    /// footer duplicates what is already accumulated; its lowest `offset`
    /// content rows — the band ending just above the footer — are new. The
    /// split sits at the verifier-gated offset; no seam search is re-run here
    /// (the reference's `seam::overlap_seam_cost` only guards against ±1px
    /// rounding, which the verifier already covers).
    pub fn append(&mut self, frame: RgbaImage, decision: &FrameDecision, offset: i64) -> u32 {
        let FrameDecision::Append { .. } = decision else {
            return 0;
        };
        self.update_sticky_top(&frame);
        self.update_sticky_bottom(&frame);
        self.update_fixed_columns(&frame);
        self.strip_base_footer();

        let fh = frame.height() as i64;
        // The fixed bottom bar is chrome, never new content: the copy stops
        // at it (provisionally at this frame's trailing-equal depth).
        let content_end = fh - self.last_bottom_depth as i64;
        // Rows of the frame that re-show existing content.
        let y_from = (content_end - offset).max(0);
        let add = (content_end - y_from).max(0) as u32;

        if add == 0 {
            self.last_frame = frame.clone();
            self.push_ring(frame);
            return 0;
        }
        let new_part = Self::rows(&frame, y_from as u32, content_end as u32);
        let mut grown = RgbaImage::new(self.image.width(), self.image.height() + add);
        imageops::overlay(&mut grown, &self.image, 0, 0);
        imageops::overlay(&mut grown, &new_part, 0, self.image.height() as i64);
        self.image = grown;
        self.clear_fixed_columns(self.base_height);
        self.history.push(offset);
        self.appended += 1;
        self.last_frame = frame.clone();
        self.push_ring(frame);
        add
    }

    fn push_ring(&mut self, frame: RgbaImage) {
        if self.ring.len() == RING_LEN {
            self.ring.pop_front();
        }
        self.ring.push_back(frame);
    }

    /// The last appended frame — the alignment target for the next capture.
    pub fn last_frame(&self) -> &RgbaImage {
        &self.last_frame
    }

    /// A downscaled snapshot of the ACCUMULATED image for the live side
    /// preview (reference `ScrollScreenshot/screen_scroll.png`: the panel
    /// watches the stitched result grow, not the newest viewport).
    ///
    /// Publishing the full-resolution buffer every ~20px step would re-encode
    /// a multi-megabyte PNG inside the capture loop, so the snapshot is capped
    /// at 900 on the longest side — far above what the ≤360×520 panel ever
    /// shows, while keeping the true aspect ratio.
    pub fn preview_image(&self) -> RgbaImage {
        crate::output::thumbnail(&self.image, 900)
    }

    /// Strip the detected sticky bars from the final image (reference
    /// `stitch_incremental_with_fixed_margins` de-duplicates fixed bars).
    pub fn finish(mut self) -> RgbaImage {
        if self.sticky_top > 0 {
            let h = self.image.height();
            let keep = h.saturating_sub(self.sticky_top);
            if keep > 0 {
                self.image = Self::rows(&self.image, self.sticky_top, h);
                log::info!(
                    "[长截图] 去除固定顶栏 {}px，最终 {}x{}",
                    self.sticky_top,
                    self.image.width(),
                    self.image.height()
                );
            }
        }
        // A confirmed footer still sitting at the image tail (no append ever
        // followed its confirmation — e.g. the session ended right there).
        if self.sticky_bottom > 0 && !self.bottom_stripped {
            let h = self.image.height();
            let keep = h.saturating_sub(self.sticky_bottom);
            if keep > 0 {
                self.image = Self::rows(&self.image, 0, keep);
                log::info!(
                    "[长截图] 去除固定底栏 {}px，最终 {}x{}",
                    self.sticky_bottom,
                    self.image.width(),
                    self.image.height()
                );
            }
        }
        self.image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(seed: u64, w: u32, h: u32) -> RgbaImage {
        use std::cell::Cell;
        let state =
            Cell::new(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        let next = || {
            let x = state.get();
            state.set(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
            (x >> 33) as u8
        };
        let mut img = RgbaImage::new(w, h);
        for y in 0..h {
            // Same generator as the pipeline tests: seed-spaced, seed-phased
            // bands so a wrong alignment can never look like a match.
            let k = (y / 7) % 256;
            let band = (k
                .wrapping_mul(23 + (seed % 13) as u32)
                .wrapping_add((seed % 256) as u32 * 37)
                % 256) as u8;
            for x in 0..w {
                let n = if x % 5 == 0 { next() / 8 } else { 0 };
                let v = band.wrapping_add(n);
                img.put_pixel(x, y, image::Rgba([v, v.wrapping_add(2), 255u8.wrapping_sub(v), 255]));
            }
        }
        img
    }

    fn window_of(src: &RgbaImage, y: u32, h: u32) -> RgbaImage {
        imageops::crop_imm(src, 0, y, src.width(), h).to_image()
    }

    fn fixed_sidebar_frame(seed: u8) -> RgbaImage {
        let mut frame = RgbaImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let value = if x < 32 {
                    ((x * 13 + y * 7) % 256) as u8
                } else {
                    seed.wrapping_mul(40).wrapping_add((x * 3 + y * 5) as u8)
                };
                frame.put_pixel(x, y, image::Rgba([value, value, value, 255]));
            }
        }
        frame
    }

    #[test]
    fn fixed_sidebar_is_not_repeated_below_the_first_viewport() {
        let mut stitcher = Stitcher::new(fixed_sidebar_frame(1));
        let decision = FrameDecision::Append {
            offset: 10,
            confidence: 1.0,
        };
        for seed in [2, 3, 4] {
            stitcher.append(fixed_sidebar_frame(seed), &decision, 10);
        }
        assert_eq!(stitcher.image.get_pixel(16, 99).0[3], 255);
        assert_eq!(stitcher.image.get_pixel(16, 100).0[3], 0);
        assert_eq!(stitcher.image.get_pixel(80, 100).0[3], 255);
    }

    #[test]
    fn stitches_three_windows() {
        let src = page(31, 200, 1400);
        let mut stitcher = Stitcher::new(window_of(&src, 0, 300));
        let mut total = 300u32;
        let mut y = 0u32;
        for &off in &[120i64, 90, 150] {
            y += off as u32;
            let frame = window_of(&src, y, 300);
            let decision = crate::scroll::pipeline::decide(stitcher.last_frame(), &frame, 12);
            let FrameDecision::Append { offset, .. } = decision else {
                panic!("expected append, got {decision:?}");
            };
            let added = stitcher.append(frame, &decision, offset);
            total += added;
            assert_eq!(stitcher.height(), total);
        }
        let out = stitcher.finish();
        assert_eq!(out.height(), total);
        // Content check: the stitched image equals the source slice.
        let expect = window_of(&src, 0, total);
        for y in (0..total).step_by(7) {
            for x in (0..200).step_by(11) {
                assert_eq!(out.get_pixel(x, y), expect.get_pixel(x, y), "at {x},{y}");
            }
        }
    }

    #[test]
    fn sticky_header_is_detected_and_stripped() {
        let src = page(41, 200, 1600);
        // Fixed 40px header prepended to every "frame".
        let header = window_of(&src, 0, 40);
        let compose = |y: u32| {
            let body = window_of(&src, 40 + y, 300);
            let mut img = RgbaImage::new(200, 340);
            imageops::overlay(&mut img, &header, 0, 0);
            imageops::overlay(&mut img, &body, 0, 40);
            img
        };
        let mut stitcher = Stitcher::new(compose(0));
        let mut y = 0u32;
        for &off in &[100i64, 110, 90] {
            y += off as u32;
            let frame = compose(y);
            let decision = crate::scroll::pipeline::decide(stitcher.last_frame(), &frame, 12);
            let FrameDecision::Append { offset, .. } = decision else {
                panic!("expected append, got {decision:?}")
            };
            stitcher.append(frame, &decision, offset);
        }
        let out = stitcher.finish();
        // The 40px header should have been detected and removed.
        assert!(out.height() < 340 + 300, "header not stripped: {}", out.height());
    }

    #[test]
    fn sticky_footer_is_trimmed_and_base_stripped() {
        let src = page(51, 200, 1600);
        // Fixed 36px footer under a 300px scrolling body.
        let footer = window_of(&src, 0, 36);
        let compose = |y: u32| {
            let body = window_of(&src, 40 + y, 300);
            let mut img = RgbaImage::new(200, 336);
            imageops::overlay(&mut img, &body, 0, 0);
            imageops::overlay(&mut img, &footer, 0, 300);
            img
        };
        let mut stitcher = Stitcher::new(compose(0));
        let mut y = 0u32;
        for &off in &[100i64, 110, 90, 120] {
            y += off as u32;
            let frame = compose(y);
            let decision = crate::scroll::pipeline::decide(stitcher.last_frame(), &frame, 12);
            let FrameDecision::Append { offset, .. } = decision else {
                panic!("expected append, got {decision:?}")
            };
            stitcher.append(frame, &decision, offset);
        }
        let out = stitcher.finish();
        // Body only: the base frame's 300 body rows plus one offset-sized
        // band per append. The footer appears nowhere — naive stitching
        // would repeat it inside every appended band (336 + 420 = 756).
        let expect_h = 300 + 100 + 110 + 90 + 120;
        assert_eq!(out.height(), expect_h, "footer not handled");
        let expect = window_of(&src, 40, expect_h);
        for y in (0..expect_h).step_by(7) {
            for x in (0..200).step_by(11) {
                assert_eq!(out.get_pixel(x, y), expect.get_pixel(x, y), "at {x},{y}");
            }
        }
    }

    #[test]
    fn ring_exposes_older_baselines() {
        let src = page(61, 120, 900);
        let mut stitcher = Stitcher::new(window_of(&src, 0, 240));
        assert!(stitcher.rescue_baseline(1).is_none());
        let mut y = 0u32;
        for &off in &[40i64, 50, 60] {
            y += off as u32;
            let frame = window_of(&src, y, 240);
            let decision = crate::scroll::pipeline::decide(stitcher.last_frame(), &frame, 12);
            let FrameDecision::Append { offset, .. } = decision else {
                panic!("expected append, got {decision:?}")
            };
            stitcher.append(frame, &decision, offset);
        }
        // rescue_baseline(1) is the frame one generation before the newest —
        // here the window taken after the first two offsets (40 + 50).
        let older = stitcher.rescue_baseline(1).unwrap();
        let expect = window_of(&src, 90, 240);
        assert_eq!(older.get_pixel(60, 60), expect.get_pixel(60, 60));
        assert_eq!(stitcher.last_committed(), Some(60));
    }
}
