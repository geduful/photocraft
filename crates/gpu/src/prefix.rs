//! One lower-stack composite, held in bounded chunk textures. Exact COW snapshots deliberately
//! invalidate on more than rendering needs: completeness matters more than a clever render key.
//!
//! Entries are **whole grid chunks** (chunk-aligned, chunk-sized document rectangles), not the
//! damage rectangles the render happens to need: a brush stroke paints a new partial rectangle
//! every frame, and caching those would fill the budget with entries that never hit again.
//! A partial rectangle is served as a sub-rect copy of the whole chunk it falls into, and a
//! capture merges into the chunk it overlaps, so every hit lands on the same stable entry.
use std::collections::HashMap;

use photocraft_doc::{Document, Layer, LayerContent, LayerMask, VectorMask, Effects, AdvancedBlending, BlendIf, Locks};
use photocraft_geom::Rect;

use crate::Tex;
use crate::plan::{Checkpoint, Plan};

/// Compare two layers by their pixel-content identity (ignoring compositing properties).
/// Returns true if the layers would produce identical pixel output when composited with
/// the same backdrop. Used by the prefix cache to find valid checkpoints when only
/// compositing properties (opacity, fill, blend, visible, etc.) change.
fn layers_pixel_equal(a: &Layer, b: &Layer) -> bool {
    a.id == b.id
        && a.content == b.content
        && a.mask == b.mask
        && a.vector_mask == b.vector_mask
        && a.effects == b.effects
        && a.clipped == b.clipped
        && a.excluded_channels == b.excluded_channels
        && a.blend_if == b.blend_if
        && a.advanced == b.advanced
        // Note: locks.transparency/position don't affect pixel output.
        // visible, opacity, fill_opacity, blend, name, label, psd_blocks, etc. are compositing properties.
}

pub(crate) const MAX_BYTES: u64 = 512 << 20;

/// One whole grid chunk, with the document rectangles that hold valid pixels: a whole-chunk
/// capture covers the cell in one entry, partial captures accumulate alongside each other.
pub(crate) struct Chunk {
    tex: Tex,
    covered: Vec<Rect>,
}

pub(crate) struct Prefix {
    context: Option<Document>,
    layers: Vec<Layer>,
    checkpoint: Option<Checkpoint>,
    chunks: HashMap<Rect, Chunk>,
    bytes: u64,
    budget: u64,
    step: i32,
}

impl Default for Prefix {
    fn default() -> Self {
        Self { context: None, layers: Vec::new(), checkpoint: None, chunks: HashMap::new(), bytes: 0, budget: MAX_BYTES, step: 512 }
    }
}

/// The whole grid chunk a rectangle falls into (chunk-aligned origin, full chunk size).
fn cell_of(rect: Rect, step: i32) -> Rect {
    let x0 = rect.x0.div_euclid(step) * step;
    let y0 = rect.y0.div_euclid(step) * step;
    Rect::new(x0, y0, x0.saturating_add(step), y0.saturating_add(step))
}

impl Prefix {
    pub(crate) fn clear(&mut self) {
        self.context = None;
        self.layers.clear();
        self.checkpoint = None;
        self.chunks.clear();
        self.bytes = 0;
    }

    pub(crate) fn forget(&mut self, doc: photocraft_doc::DocId) {
        if self.context.as_ref().is_some_and(|d| d.id == doc) {
            self.clear();
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn budget(&self) -> u64 {
        self.budget
    }

    pub(crate) fn set_budget(&mut self, bytes: u64) {
        self.budget = bytes.min(MAX_BYTES);
        if self.bytes > self.budget || self.budget == 0 {
            self.clear();
        }
    }

    /// The grid step the entries are aligned to; the compositor reports its chunk size.
    pub(crate) fn set_step(&mut self, step: u32) {
        let step = step.max(1) as i32;
        if step != self.step {
            self.step = step;
            self.clear();
        }
    }

    /// The document the cached composite belongs to, if any.
    pub(crate) fn doc(&self) -> Option<photocraft_doc::DocId> {
        self.context.as_ref().map(|d| d.id)
    }

    pub(crate) fn prepare(&mut self, doc: &Document, plan: &Plan<'_>) -> Option<Checkpoint> {
        if self.budget == 0 {
            return None;
        }
        // Keeping every non-layer field makes new document inputs invalidate automatically.
        // Tile data is Arc-shared; shared Tile equality does not scan its pixel bytes.
        let mut context = doc.clone();
        let mut layers = std::mem::take(&mut context.layers);
        let same_context = self.context.as_ref() == Some(&context);

        // Find first layer where PIXEL CONTENT changed (content, mask, effects, etc.).
        let pixel_changed = self.layers.iter().zip(&layers).position(|(a, b)| !layers_pixel_equal(a, b))
            .or_else(|| (layers.len() < self.layers.len()).then_some(layers.len()));
        // Find first layer where ANY property changed (including compositing: opacity, fill, blend, visible).
        let any_changed = self.layers.iter().zip(&layers).position(|(a, b)| a != b)
            .or_else(|| (layers.len() < self.layers.len()).then_some(layers.len()));

        // The prefix captures layers 0..layer_end-1. Both pixel and compositing changes on layer N
        // require layer_end <= N. Use the earliest change.
        let changed = match (pixel_changed, any_changed) {
            (Some(p), Some(a)) => Some(p.min(a)),
            (Some(p), None) => Some(p),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        };

        let old = self.checkpoint.filter(|c| same_context && changed.is_none() && plan.checkpoints.contains(c));
        let checkpoint = old.or_else(|| {
            if same_context && let Some(changed) = changed {
                plan.checkpoints.iter().rev().find(|c| c.layer_end <= changed && c.pass_end > 1).copied()
            } else {
                let mut useful = plan.checkpoints.iter().rev().filter(|c| c.pass_end > 1);
                let last = useful.next().copied();
                useful.next().copied().or(last)
            }
        });
        if old.is_none() {
            self.clear();
        }
        let Some(checkpoint) = checkpoint else {
            // Remember bottom-layer edits even when no useful prefix survives: otherwise
            // every other edit would populate and immediately discard the default checkpoint.
            self.context = Some(context);
            self.layers = layers;
            return None;
        };
        layers.truncate(checkpoint.layer_end);
        self.context = Some(context);
        self.layers = layers;
        self.checkpoint = Some(checkpoint);
        Some(checkpoint)
    }

    /// Whether every pixel of `rect` is already cached in its whole grid chunk.
    pub(crate) fn contains(&self, rect: Rect) -> bool {
        self.covered_by(rect).is_some()
    }

    /// The whole chunk and the source offset that serve `rect`, when it is fully covered.
    fn covered_by(&self, rect: Rect) -> Option<(&Chunk, (i32, i32))> {
        let cell = cell_of(rect, self.step);
        let chunk = self.chunks.get(&cell)?;
        let offset = (rect.x0 - cell.x0, rect.y0 - cell.y0);
        chunk.covered.iter().any(|c| c.contains_rect(&rect)).then_some((chunk, offset))
    }

    /// Copies the cached pixels of `rect` into `dst` (its accumulator slot). Only whole-chunk
    /// and sub-rect copies out of the whole chunk; a partial hit is not pretended to be full.
    pub(crate) fn copy_out(&self, encoder: &mut wgpu::CommandEncoder, rect: Rect, dst: &wgpu::Texture) -> bool {
        let Some((chunk, (ox, oy))) = self.covered_by(rect) else { return false };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &chunk.tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: ox.unsigned_abs(), y: oy.unsigned_abs(), z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo { texture: dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: rect.width(), height: rect.height(), depth_or_array_layers: 1 },
        );
        true
    }

    /// Merges the rendered `rect` (document coordinates) into its whole grid chunk, admitting
    /// the chunk while space remains. Replacing entries on every miss would make a full
    /// refresh larger than the cache evict its own next hits.
    pub(crate) fn capture(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, rect: Rect, src: &wgpu::Texture) {
        if rect.is_empty() {
            return;
        }
        let cell = cell_of(rect, self.step);
        let bpp = if src.format() == wgpu::TextureFormat::Rgba32Float { 16 } else { 8 };
        let cell_bytes = u64::from(cell.width()).saturating_mul(u64::from(cell.height())).saturating_mul(bpp);
        if !self.chunks.contains_key(&cell) {
            if cell_bytes == 0 || cell_bytes > self.budget.saturating_sub(self.bytes) {
                return;
            }
            let tex =
                Tex::new(device, "pc_compose_prefix", cell.width(), cell.height(), src.format(), wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
            self.bytes = self.bytes.saturating_add(cell_bytes);
            self.chunks.insert(cell, Chunk { tex, covered: Vec::new() });
        }
        let Some(chunk) = self.chunks.get_mut(&cell) else { return };
        // A whole-chunk render replaces the coverage; a partial rectangle lands at its offset
        // inside the cell, beside whatever earlier strokes put there.
        if rect == cell {
            chunk.covered.clear();
        } else if chunk.covered.iter().any(|c| c.contains_rect(&rect)) {
            return; // already cached from an earlier frame
        }
        let (ox, oy) = (rect.x0 - cell.x0, rect.y0 - cell.y0);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo {
                texture: &chunk.tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: ox.unsigned_abs(), y: oy.unsigned_abs(), z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: rect.width(), height: rect.height(), depth_or_array_layers: 1 },
        );
        chunk.covered.push(rect);
        if chunk.covered.iter().any(|c| c == &cell) {
            chunk.covered = vec![cell];
        }
    }
}
