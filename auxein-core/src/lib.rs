#![forbid(unsafe_code)]

//! Auxein v0.4.0 production core.
//!
//! The crate is deliberately dependency-free. Persistent geometry is stored
//! in the selected scalar (`f32` or `f64`), every cognitive calculation is
//! performed in `f64`, and material accounting is exact integer arithmetic.

mod decimal;
mod json;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::mem;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering},
    Arc,
};

pub const FORMAT_VERSION: u64 = 4;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Geometry,
    Temporal,
    Predictive,
}

impl Mode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "geometry" => Ok(Self::Geometry),
            "temporal" => Ok(Self::Temporal),
            "predictive" => Ok(Self::Predictive),
            _ => Err(Error::Invalid(
                "mode must be 'geometry', 'temporal' or 'predictive'".into(),
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Temporal => "temporal",
            Self::Predictive => "predictive",
        }
    }

    pub const fn temporal(self) -> bool {
        matches!(self, Self::Temporal | Self::Predictive)
    }

    pub const fn predictive(self) -> bool {
        matches!(self, Self::Predictive)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Space {
    Geometry,
    Temporal,
}

impl Space {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Temporal => "temporal",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Json(String),
    Io(String),
    Inexecutable(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid Auxein input: {s}"),
            Self::Json(s) => write!(f, "invalid JSON: {s}"),
            Self::Io(s) => write!(f, "I/O error: {s}"),
            Self::Inexecutable(s) => write!(f, "inexecutable Auxein state: {s}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Budget {
    /// Exact raw material units.
    Units(u64),
    /// Ergonomic kernel capacity, parsed as an exact decimal.
    Kernels(String),
}

impl Budget {
    pub fn kernels(value: impl ToString) -> Self {
        Self::Kernels(value.to_string())
    }

    pub const fn units(value: u64) -> Self {
        Self::Units(value)
    }
}

mod sealed {
    pub trait Sealed {
        fn from_finite(value: f64) -> Self;
    }

    impl Sealed for f32 {
        fn from_finite(value: f64) -> Self {
            let out = value as f32;
            if out == 0.0 {
                0.0
            } else {
                out
            }
        }
    }

    impl Sealed for f64 {
        fn from_finite(value: f64) -> Self {
            if value == 0.0 {
                0.0
            } else {
                value
            }
        }
    }
}

/// Persistent scalar supported by the canonical engine.
pub trait Scalar:
    sealed::Sealed + Copy + Clone + fmt::Debug + PartialEq + Send + Sync + 'static
{
    const NAME: &'static str;
    const BYTES: u64;
    fn from_f64(value: f64) -> Result<Self>;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
    fn min_positive() -> Self;
}

impl Scalar for f32 {
    const NAME: &'static str = "f32";
    const BYTES: u64 = 4;

    fn from_f64(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::Invalid("persistent real must be finite".into()));
        }
        let out = value as f32;
        if !out.is_finite() {
            return Err(Error::Invalid("persistent f32 overflow".into()));
        }
        Ok(if out == 0.0 { 0.0 } else { out })
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn bits(self) -> u64 {
        self.to_bits() as u64
    }

    fn min_positive() -> Self {
        f32::from_bits(1)
    }
}

impl Scalar for f64 {
    const NAME: &'static str = "f64";
    const BYTES: u64 = 8;

    fn from_f64(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::Invalid("persistent real must be finite".into()));
        }
        Ok(if value == 0.0 { 0.0 } else { value })
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn bits(self) -> u64 {
        self.to_bits()
    }

    fn min_positive() -> Self {
        f64::from_bits(1)
    }
}

// CELL support decays homothetically and has no authority while the CELL is
// outside the concern set.  The layer clock lets untouched CELLs defer those
// exact scalar projections until their support is observed again.  Changing
// eta materializes every pending decay before installing a new clock, so a
// kernel never spans two lambda values.
#[derive(Debug)]
struct DecayClock {
    epoch: AtomicU32,
    lambda_bits: AtomicU64,
}

impl DecayClock {
    fn new(epoch: u32, lambda: f64) -> Self {
        Self {
            epoch: AtomicU32::new(epoch),
            lambda_bits: AtomicU64::new(lambda.to_bits()),
        }
    }

    #[inline]
    fn epoch(&self) -> u32 {
        self.epoch.load(AtomicOrdering::Relaxed)
    }

    #[inline]
    fn lambda(&self) -> f64 {
        f64::from_bits(self.lambda_bits.load(AtomicOrdering::Relaxed))
    }

    #[inline]
    fn reset(&self, lambda: f64) {
        self.epoch.store(0, AtomicOrdering::Relaxed);
        self.lambda_bits
            .store(lambda.to_bits(), AtomicOrdering::Relaxed);
    }
}

#[derive(Clone, Debug)]
pub struct Kernel<S: Scalar> {
    weight: S,
    center: Vec<S>,
    variance: S,
    norm2: f64,
    dirty: bool,
    decay_clock: Option<Arc<DecayClock>>,
    decay_epoch: u32,
}

impl<S: Scalar> Kernel<S> {
    pub fn new(weight: f64, center: &[f64], variance: f64) -> Result<Self> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Error::Invalid(
                "kernel weight must be finite and positive".into(),
            ));
        }
        if !variance.is_finite() || variance < 0.0 {
            return Err(Error::Invalid(
                "kernel variance must be finite and nonnegative".into(),
            ));
        }
        if center.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("kernel center must be finite".into()));
        }
        project_kernel::<S>(Kernel64 {
            weight,
            center: center.to_vec(),
            variance,
        })
    }

    pub fn weight(&self) -> f64 {
        if let Some(clock) = &self.decay_clock {
            decayed_weight::<S>(self.weight, self.decay_epoch, clock.epoch(), clock.lambda())
                .to_f64()
        } else {
            self.weight.to_f64()
        }
    }

    #[inline]
    fn materialize_weight_at(&mut self, epoch: u32, lambda: f64) -> f64 {
        if self.decay_epoch != epoch {
            self.weight = decayed_weight::<S>(self.weight, self.decay_epoch, epoch, lambda);
            self.decay_epoch = epoch;
        }
        self.weight.to_f64()
    }

    fn bind_decay_clock(&mut self, clock: Arc<DecayClock>, epoch: u32) {
        self.decay_clock = Some(clock);
        self.decay_epoch = epoch;
    }

    pub fn center(&self) -> Vec<f64> {
        self.center.iter().map(|&x| x.to_f64()).collect()
    }

    pub fn variance(&self) -> f64 {
        self.variance.to_f64()
    }

    pub fn energy(&self) -> f64 {
        self.weight() * self.variance()
    }
}

impl<S: Scalar> PartialEq for Kernel<S> {
    fn eq(&self, other: &Self) -> bool {
        self.weight() == other.weight()
            && self.center == other.center
            && self.variance == other.variance
    }
}

#[derive(Debug)]
pub struct Layer<S: Scalar> {
    sigma: Vec<Kernel<S>>,
    cells: Vec<Kernel<S>>,
    cell_decay: Arc<DecayClock>,
    temporal_sigma: Vec<Kernel<S>>,
    temporal_cells: Vec<Kernel<S>>,
    temporal_decay: Arc<DecayClock>,
    previous: Option<Kernel<S>>,
}

impl<S: Scalar> PartialEq for Layer<S> {
    fn eq(&self, other: &Self) -> bool {
        self.sigma == other.sigma
            && self.cells == other.cells
            && self.temporal_sigma == other.temporal_sigma
            && self.temporal_cells == other.temporal_cells
            && self.previous == other.previous
    }
}

impl<S: Scalar> Clone for Layer<S> {
    fn clone(&self) -> Self {
        let epoch = self.cell_decay.epoch();
        let clock = Arc::new(DecayClock::new(epoch, self.cell_decay.lambda()));
        let mut cells = self.cells.clone();
        for cell in &mut cells {
            cell.bind_decay_clock(clock.clone(), cell.decay_epoch);
        }
        let temporal_epoch = self.temporal_decay.epoch();
        let temporal_clock = Arc::new(DecayClock::new(
            temporal_epoch,
            self.temporal_decay.lambda(),
        ));
        let mut temporal_cells = self.temporal_cells.clone();
        for cell in &mut temporal_cells {
            cell.bind_decay_clock(temporal_clock.clone(), cell.decay_epoch);
        }
        Self {
            sigma: self.sigma.clone(),
            cells,
            cell_decay: clock,
            temporal_sigma: self.temporal_sigma.clone(),
            temporal_cells,
            temporal_decay: temporal_clock,
            previous: self.previous.clone(),
        }
    }
}

impl<S: Scalar> Layer<S> {
    pub fn cells(&self) -> &[Kernel<S>] {
        &self.cells
    }

    pub fn sigma(&self) -> &[Kernel<S>] {
        &self.sigma
    }

    pub fn temporal_cells(&self) -> &[Kernel<S>] {
        &self.temporal_cells
    }

    pub fn temporal_sigma(&self) -> &[Kernel<S>] {
        &self.temporal_sigma
    }

    pub fn previous(&self) -> Option<&Kernel<S>> {
        self.previous.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub universe: Arc<str>,
    pub local_input: Arc<[f64]>,
    pub recognised: Vec<f64>,
}

impl Recognition {
    pub fn universe(&self) -> &str {
        &self.universe
    }

    pub fn local_input(&self) -> &[f64] {
        &self.local_input
    }

    pub fn recognised(&self) -> &[f64] {
        &self.recognised
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRecognition {
    pub universe: Arc<str>,
    pub previous_input: Arc<[f64]>,
    pub current_input: Arc<[f64]>,
    pub previous_recognised: Vec<f64>,
    pub current_recognised: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub universe: Arc<str>,
    pub current_context: Vec<f64>,
    pub recognised_source: Vec<f64>,
    pub predicted_successor: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Readout {
    Geometry(Vec<Recognition>),
    Temporal {
        concepts: Vec<Recognition>,
        sequences: Vec<TemporalRecognition>,
    },
    Predictive {
        concepts: Vec<Recognition>,
        sequences: Vec<TemporalRecognition>,
        predictions: Vec<Prediction>,
    },
}

impl Readout {
    pub fn concepts(&self) -> &[Recognition] {
        match self {
            Self::Geometry(values) => values,
            Self::Temporal { concepts, .. } | Self::Predictive { concepts, .. } => concepts,
        }
    }

    pub fn sequences(&self) -> &[TemporalRecognition] {
        match self {
            Self::Geometry(_) => &[],
            Self::Temporal { sequences, .. } | Self::Predictive { sequences, .. } => sequences,
        }
    }

    pub fn predictions(&self) -> &[Prediction] {
        match self {
            Self::Predictive { predictions, .. } => predictions,
            Self::Geometry(_) | Self::Temporal { .. } => &[],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.concepts().is_empty() && self.sequences().is_empty() && self.predictions().is_empty()
    }

    /// Number of conceptual recognitions. Geometry-mode callers can keep the
    /// flat readout ergonomics while temporal callers inspect
    /// `sequences()` explicitly.
    pub fn len(&self) -> usize {
        self.concepts().len()
    }
}

impl std::ops::Index<usize> for Readout {
    type Output = Recognition;

    fn index(&self, index: usize) -> &Self::Output {
        &self.concepts()[index]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Transformation {
    ClearSigma {
        count: usize,
    },
    TrimLayers {
        count: usize,
    },
    DestroyCells {
        count: usize,
        waves: usize,
        k_through: f64,
    },
    Promote {
        space: Space,
        layer: usize,
        count: usize,
    },
    GrowthCommit {
        geometric_seeds: usize,
        temporal_seeds: usize,
        seeds: usize,
        layer_created: bool,
        units: u64,
    },
    GrowthReject {
        geometric_seeds: usize,
        temporal_seeds: usize,
        seeds: usize,
        layer_requested: bool,
        units: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerReport {
    pub phase: Space,
    pub layer_index: usize,
    pub input_atom_count: usize,
    pub input_mass: f64,
    pub unknown_atom_count: usize,
    pub recognised_atom_count: usize,
    pub cell_count_before: usize,
    pub cell_count_after: usize,
    pub sigma_count_before: usize,
    pub sigma_count_after: usize,
    pub promoted: usize,
    pub seed_requests: usize,
    pub context_emitted: bool,
    pub output_atom_count: usize,
    pub output_mass: f64,
    pub context_center: Option<Vec<f64>>,
    pub context_variance: Option<f64>,
    pub recognition_count: usize,
    pub cell_responsibility_mass: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    pub step_index: u64,
    pub readout: Readout,
    pub transformations: Vec<Transformation>,
    pub maintenance_open_units: u64,
    pub maintenance_units: u64,
    pub budget_units: u64,
    pub layer_reports: Vec<LayerReport>,
    pub temporal_reports: Vec<LayerReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub steps_seen: u64,
    pub dimension: usize,
    pub universe: String,
    pub scalar: &'static str,
    pub memory: f64,
    pub eta: f64,
    pub mode: Mode,
    pub chi: f64,
    pub alpha: f64,
    pub effective_alpha: f64,
    pub layer_count: usize,
    pub cells_per_layer: Vec<usize>,
    pub sigma_per_layer: Vec<usize>,
    pub temporal_cells_per_layer: Vec<usize>,
    pub temporal_sigma_per_layer: Vec<usize>,
    pub previous_context_per_layer: Vec<bool>,
    pub maintenance_units: u64,
    pub budget: String,
    pub budget_units: u64,
    pub budget_margin_units: i128,
    pub is_solvent: bool,
}

#[derive(Debug)]
pub struct Auxein<S: Scalar> {
    dimension: usize,
    memory: S,
    eta: S,
    universe: Arc<str>,
    mode: Mode,
    steps_seen: u64,
    layers: Vec<Layer<S>>,
    chi: f64,
    alpha: f64,
    beta: f64,
    lambda: f64,
    budget_units: u64,
    scratch_targets: Targets,
    scratch_concerned: Vec<(usize, f64)>,
    scratch_unknown: Vec<usize>,
}

impl<S: Scalar> Clone for Auxein<S> {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            memory: self.memory,
            eta: self.eta,
            universe: self.universe.clone(),
            mode: self.mode,
            steps_seen: self.steps_seen,
            layers: self.layers.clone(),
            chi: self.chi,
            alpha: self.alpha,
            beta: self.beta,
            lambda: self.lambda,
            budget_units: self.budget_units,
            scratch_targets: Targets::default(),
            scratch_concerned: Vec::new(),
            scratch_unknown: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Network {
    F32(Auxein<f32>),
    F64(Auxein<f64>),
}

pub type AuxeinF32 = Auxein<f32>;
pub type AuxeinF64 = Auxein<f64>;

#[derive(Clone, Debug)]
struct Atom {
    x: Arc<[f64]>,
    r: f64,
    variance: f64,
    norm2: f64,
    zero: bool,
}

#[derive(Clone, Debug)]
struct Kernel64 {
    weight: f64,
    center: Vec<f64>,
    variance: f64,
}

#[derive(Clone, Debug)]
struct TargetContribution {
    atom_index: usize,
    weight: f64,
}

#[derive(Clone, Debug, Default)]
struct Targets {
    weights: Vec<f64>,
    centers: Vec<f64>,
    variances: Vec<f64>,
    touched: Vec<usize>,
    changed: Vec<usize>,
    dimension: usize,
    batches: Vec<Vec<TargetContribution>>,
    sum_terms: Vec<f64>,
    variance_terms: Vec<f64>,
    sum_partials: Vec<f64>,
}

impl Targets {
    fn reset(&mut self, count: usize, dimension: usize, need_centers: bool) {
        for index in self.touched.drain(..) {
            self.weights[index] = 0.0;
            if index < self.batches.len() {
                self.batches[index].clear();
            }
        }
        self.dimension = dimension;
        if self.weights.len() < count {
            self.weights.resize(count, 0.0);
            self.variances.resize(count, 0.0);
        }
        if self.batches.len() < count {
            self.batches.resize_with(count, Vec::new);
        }
        if need_centers {
            let center_count = count.saturating_mul(dimension);
            if self.centers.len() < center_count {
                self.centers.resize(center_count, 0.0);
            }
        }
    }

    fn mark_single(&mut self, index: usize, distance2: f64) {
        debug_assert_eq!(self.weights[index], 0.0);
        self.touched.push(index);
        self.variances[index] = distance2;
    }

    fn set_single_weight(&mut self, index: usize, weight: f64) {
        debug_assert!(weight > 0.0);
        debug_assert_eq!(self.weights[index], 0.0);
        self.weights[index] = weight;
    }

    fn add_atom(&mut self, index: usize, atom_index: usize, weight: f64) {
        if weight <= 0.0 {
            return;
        }
        if self.batches[index].is_empty() {
            self.touched.push(index);
        }
        self.batches[index].push(TargetContribution { atom_index, weight });
    }

    fn finalize_batches(&mut self, presentation: &[Atom]) {
        for touched_pos in 0..self.touched.len() {
            let index = self.touched[touched_pos];
            let batch = &self.batches[index];
            if batch.is_empty() {
                continue;
            }

            let start = index * self.dimension;
            let end = start + self.dimension;
            if batch.len() == 1 {
                let item = &batch[0];
                let atom = &presentation[item.atom_index];
                self.weights[index] = item.weight;
                self.centers[start..end].copy_from_slice(&atom.x);
                self.variances[index] = atom.variance;
                continue;
            }

            let first_atom = &presentation[batch[0].atom_index];
            if batch[1..].iter().all(|item| {
                let atom = &presentation[item.atom_index];
                atom.x == first_atom.x && atom.variance == first_atom.variance
            }) {
                self.sum_terms.clear();
                self.sum_terms.extend(batch.iter().map(|item| item.weight));
                self.weights[index] = orderless_sum(&self.sum_terms, &mut self.sum_partials);
                self.centers[start..end].copy_from_slice(&first_atom.x);
                self.variances[index] = first_atom.variance;
                continue;
            }

            self.sum_terms.clear();
            self.sum_terms.extend(batch.iter().map(|item| item.weight));
            let weight = orderless_sum(&self.sum_terms, &mut self.sum_partials);
            self.weights[index] = weight;

            for dimension_index in 0..self.dimension {
                self.sum_terms.clear();
                self.sum_terms.extend(
                    batch
                        .iter()
                        .map(|item| item.weight * presentation[item.atom_index].x[dimension_index]),
                );
                self.centers[start + dimension_index] = structural_zero(
                    orderless_sum(&self.sum_terms, &mut self.sum_partials) / weight,
                );
            }

            self.variance_terms.clear();
            for item in batch {
                let atom = &presentation[item.atom_index];
                self.sum_terms.clear();
                self.sum_terms
                    .extend(
                        atom.x
                            .iter()
                            .zip(&self.centers[start..end])
                            .map(|(&x, &center)| {
                                let delta = x - center;
                                delta * delta
                            }),
                    );
                let distance2 = orderless_sum(&self.sum_terms, &mut self.sum_partials);
                self.variance_terms
                    .push(item.weight * (atom.variance + distance2));
            }
            self.variances[index] = structural_zero(
                orderless_sum(&self.variance_terms, &mut self.sum_partials) / weight,
            );
        }
    }

    fn apply_population<S: Scalar>(
        &mut self,
        kernels: &mut [Kernel<S>],
        presentation: &[Atom],
        beta: f64,
        lambda: f64,
    ) -> Result<()> {
        self.finalize_batches(presentation);
        if self.touched.windows(2).any(|pair| pair[0] > pair[1]) {
            self.touched.sort_unstable();
        }
        let mut begin = 0;
        for touched_index in 0..self.touched.len() {
            let index = self.touched[touched_index];
            for kernel in &mut kernels[begin..index] {
                decay_kernel(kernel, lambda);
            }
            self.apply_touched_ema(index, &mut kernels[index], beta, lambda)?;
            if zero_scalar_vec(&kernels[index].center) {
                kernels[index].dirty = true;
            }
            begin = index + 1;
        }
        for kernel in &mut kernels[begin..] {
            decay_kernel(kernel, lambda);
        }
        Ok(())
    }

    fn apply_single_population<S: Scalar>(
        &mut self,
        kernels: &mut [Kernel<S>],
        x: &[f64],
        beta: f64,
        lambda: f64,
    ) -> Result<()> {
        debug_assert!(self.touched.windows(2).all(|pair| pair[0] < pair[1]));
        let mut begin = 0;
        for touched_index in 0..self.touched.len() {
            let index = self.touched[touched_index];
            for kernel in &mut kernels[begin..index] {
                decay_kernel(kernel, lambda);
            }
            self.apply_single_touched_ema(index, &mut kernels[index], x, beta, lambda)?;
            if zero_scalar_vec(&kernels[index].center) {
                kernels[index].dirty = true;
            }
            begin = index + 1;
        }
        for kernel in &mut kernels[begin..] {
            decay_kernel(kernel, lambda);
        }
        Ok(())
    }

    fn apply_cell_population<S: Scalar>(
        &mut self,
        kernels: &mut [Kernel<S>],
        presentation: &[Atom],
        beta: f64,
        lambda: f64,
        epoch: u32,
    ) -> Result<()> {
        self.finalize_batches(presentation);
        if self.touched.windows(2).any(|pair| pair[0] > pair[1]) {
            self.touched.sort_unstable();
        }
        for &index in &self.touched {
            self.apply_touched_ema(index, &mut kernels[index], beta, lambda)?;
            kernels[index].decay_epoch = epoch;
            if zero_scalar_vec(&kernels[index].center) {
                kernels[index].dirty = true;
            }
        }
        Ok(())
    }

    fn apply_single_cell_population<S: Scalar>(
        &mut self,
        kernels: &mut [Kernel<S>],
        x: &[f64],
        beta: f64,
        lambda: f64,
        epoch: u32,
    ) -> Result<()> {
        debug_assert!(self.touched.windows(2).all(|pair| pair[0] < pair[1]));
        for &index in &self.touched {
            self.apply_single_touched_ema(index, &mut kernels[index], x, beta, lambda)?;
            kernels[index].decay_epoch = epoch;
            if zero_scalar_vec(&kernels[index].center) {
                kernels[index].dirty = true;
            }
        }
        Ok(())
    }

    fn apply_single_touched_ema<S: Scalar>(
        &self,
        index: usize,
        old: &mut Kernel<S>,
        target_center: &[f64],
        beta: f64,
        lambda: f64,
    ) -> Result<()> {
        self.apply_ema_core(
            index,
            old,
            target_center,
            0.0,
            self.variances[index],
            beta,
            lambda,
        )?;
        Ok(())
    }

    fn apply_touched_ema<S: Scalar>(
        &self,
        index: usize,
        old: &mut Kernel<S>,
        beta: f64,
        lambda: f64,
    ) -> Result<bool> {
        let start = index * self.dimension;
        let end = start + self.dimension;
        let target_center = &self.centers[start..end];
        let delta2 = stable_sum(old.center.iter().zip(target_center).map(|(&old, &new)| {
            let d = old.to_f64() - new;
            d * d
        }));
        self.apply_ema_core(
            index,
            old,
            target_center,
            self.variances[index],
            delta2,
            beta,
            lambda,
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn apply_ema_core<S: Scalar>(
        &self,
        index: usize,
        old: &mut Kernel<S>,
        target_center: &[f64],
        target_variance: f64,
        delta2: f64,
        beta: f64,
        lambda: f64,
    ) -> Result<bool> {
        let old_weight = old.weight.to_f64();
        let a = lambda * old_weight;
        let target_weight = self.weights[index];
        debug_assert!(target_weight > 0.0);
        let b = beta * target_weight;
        let total = a + b;
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::Invalid(
                "EMA produced nonpositive or nonfinite support".into(),
            ));
        }
        let ratio = b / total;
        let mut changed = false;
        let mut norm_sum = 0.0;
        let mut norm_correction = 0.0;
        for (old_component, &new) in old.center.iter_mut().zip(target_center) {
            let old_value = old_component.to_f64();
            let projected = S::from_f64(structural_zero(old_value + ratio * (new - old_value)))?;
            changed |= projected != *old_component;
            *old_component = projected;
            let value = projected.to_f64();
            compensated_add(&mut norm_sum, &mut norm_correction, value * value);
        }
        let variance = (a * old.variance.to_f64() + b * target_variance) / total
            + (a / total) * (b / total) * delta2;
        let mut weight = S::from_f64(total)?;
        if weight.to_f64() <= 0.0 {
            weight = S::min_positive();
        }
        let variance = S::from_f64(structural_zero(variance))?;
        if variance.to_f64() < 0.0 {
            return Err(Error::Invalid(
                "persistent kernel variance is negative".into(),
            ));
        }
        changed |= variance != old.variance;
        old.weight = weight;
        old.variance = variance;
        old.norm2 = compensated_finish(norm_sum, norm_correction);
        old.dirty = changed;
        Ok(changed)
    }
}

#[inline(always)]
fn decay_weight_once<S: Scalar>(weight: S, lambda: f64) -> S {
    let mut next = <S as sealed::Sealed>::from_finite(lambda * weight.to_f64());
    if next.to_f64() <= 0.0 {
        next = S::min_positive();
    }
    next
}

#[inline]
fn decayed_weight<S: Scalar>(mut weight: S, from_epoch: u32, to_epoch: u32, lambda: f64) -> S {
    debug_assert!(from_epoch <= to_epoch);
    let age = to_epoch - from_epoch;
    if age == 0 {
        return weight;
    }
    weight = decay_weight_once(weight, lambda);
    for _ in 1..age {
        weight = decay_weight_once(weight, lambda);
    }
    weight
}

fn decay_kernel<S: Scalar>(kernel: &mut Kernel<S>, lambda: f64) {
    let mut weight = <S as sealed::Sealed>::from_finite(lambda * kernel.weight.to_f64());
    if weight.to_f64() <= 0.0 {
        weight = S::min_positive();
    }
    kernel.weight = weight;
}

struct LayerResult {
    output: Vec<Atom>,
    context: Option<Kernel64>,
    readout: Vec<Recognition>,
    seed_requests: Vec<Kernel64>,
    transformations: Vec<Transformation>,
    report: Option<LayerReport>,
}

impl<S: Scalar> Auxein<S> {
    pub fn new(
        dimension: usize,
        memory: f64,
        eta: f64,
        budget: Budget,
        universe: impl Into<String>,
    ) -> Result<Self> {
        Self::new_with_mode(dimension, memory, eta, Mode::Geometry, budget, universe)
    }

    pub fn new_with_mode(
        dimension: usize,
        memory: f64,
        eta: f64,
        mode: Mode,
        budget: Budget,
        universe: impl Into<String>,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::Invalid(
                "dimension must be a positive integer".into(),
            ));
        }
        if !memory.is_finite() || memory <= 0.0 {
            return Err(Error::Invalid("memory must be finite and positive".into()));
        }
        if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
            return Err(Error::Invalid("eta must lie in [0, 1]".into()));
        }
        let universe = universe.into();
        if universe.is_empty() {
            return Err(Error::Invalid("universe must be nonempty".into()));
        }
        let universe: Arc<str> = Arc::from(universe);
        let memory = S::from_f64(memory)?;
        let eta = S::from_f64(eta)?;
        let mut out = Self {
            dimension,
            memory,
            eta,
            universe,
            mode,
            steps_seen: 0,
            layers: vec![Layer {
                sigma: Vec::new(),
                cells: Vec::new(),
                cell_decay: Arc::new(DecayClock::new(0, 1.0)),
                temporal_sigma: Vec::new(),
                temporal_cells: Vec::new(),
                temporal_decay: Arc::new(DecayClock::new(0, 1.0)),
                previous: None,
            }],
            chi: 0.0,
            alpha: 0.0,
            beta: 0.0,
            lambda: 0.0,
            budget_units: 0,
            scratch_targets: Targets::default(),
            scratch_concerned: Vec::new(),
            scratch_unknown: Vec::new(),
        };
        out.refresh_clock();
        out.layers[0].cell_decay = Arc::new(DecayClock::new(0, out.lambda));
        out.layers[0].temporal_decay = Arc::new(DecayClock::new(0, out.lambda));
        out.budget_units = out.resolve_budget(budget)?;
        if out.budget_units < out.min_units()? {
            return Err(Error::Invalid(
                "budget is below the minimal executable state".into(),
            ));
        }
        Ok(out)
    }

    fn refresh_clock(&mut self) {
        let memory = self.memory.to_f64();
        let eta = self.eta.to_f64();
        self.chi = 2.0_f64.powf(-1.0 / memory);
        self.alpha = -(-std::f64::consts::LN_2 / memory).exp_m1();
        self.beta = eta * self.alpha;
        self.lambda = 1.0 - self.beta;
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn memory(&self) -> f64 {
        self.memory.to_f64()
    }

    pub fn eta(&self) -> f64 {
        self.eta.to_f64()
    }

    pub fn universe(&self) -> &str {
        &self.universe
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn steps_seen(&self) -> u64 {
        self.steps_seen
    }

    pub fn layers(&self) -> &[Layer<S>] {
        &self.layers
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn kernel_units(&self) -> Result<u64> {
        (self.dimension as u64)
            .checked_add(2)
            .and_then(|n| n.checked_mul(S::BYTES))
            .ok_or_else(|| Error::Invalid("dimension is too large for material accounting".into()))
    }

    pub fn temporal_kernel_units(&self) -> Result<u64> {
        (2u64)
            .checked_mul(self.dimension as u64)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(S::BYTES))
            .ok_or_else(|| Error::Invalid("dimension is too large for material accounting".into()))
    }

    pub fn network_units(&self) -> Result<u64> {
        34u64
            .checked_add(2 * S::BYTES)
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))
    }

    pub fn layer_units(&self) -> Result<u64> {
        match self.mode {
            Mode::Geometry => Ok(16),
            Mode::Temporal | Mode::Predictive => 33u64
                .checked_add(self.kernel_units()?)
                .ok_or_else(|| Error::Invalid("material accounting overflow".into())),
        }
    }

    pub fn min_units(&self) -> Result<u64> {
        self.network_units()?
            .checked_add(self.layer_units()?)
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))
    }

    fn resolve_budget(&self, budget: Budget) -> Result<u64> {
        match budget {
            Budget::Units(units) => Ok(units),
            Budget::Kernels(text) => {
                let extra = decimal::floor_mul(&text, self.kernel_units()?)?;
                self.min_units()?
                    .checked_add(extra)
                    .ok_or_else(|| Error::Invalid("budget is too large".into()))
            }
        }
    }

    pub fn budget_units(&self) -> u64 {
        self.budget_units
    }

    pub fn budget_equivalent(&self) -> Result<String> {
        ratio_string(
            self.budget_units.saturating_sub(self.min_units()?),
            self.kernel_units()?,
        )
    }

    pub fn set_budget(&mut self, budget: Budget) -> Result<()> {
        let units = self.resolve_budget(budget)?;
        if units < self.min_units()? {
            return Err(Error::Invalid(
                "budget is below the minimal executable state".into(),
            ));
        }
        self.budget_units = units;
        Ok(())
    }

    pub fn set_eta(&mut self, eta: f64) -> Result<()> {
        if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
            return Err(Error::Invalid("eta must lie in [0, 1]".into()));
        }
        let eta = S::from_f64(eta)?;
        if eta == self.eta {
            return Ok(());
        }
        for layer in &mut self.layers {
            let epoch = layer.cell_decay.epoch();
            let lambda = layer.cell_decay.lambda();
            for cell in &mut layer.cells {
                cell.materialize_weight_at(epoch, lambda);
            }
            if self.mode.temporal() {
                let epoch = layer.temporal_decay.epoch();
                let lambda = layer.temporal_decay.lambda();
                for cell in &mut layer.temporal_cells {
                    cell.materialize_weight_at(epoch, lambda);
                }
            }
        }
        self.eta = eta;
        self.refresh_clock();
        for layer in &mut self.layers {
            for cell in &mut layer.cells {
                cell.decay_epoch = 0;
            }
            layer.cell_decay.reset(self.lambda);
            if self.mode.temporal() {
                for cell in &mut layer.temporal_cells {
                    cell.decay_epoch = 0;
                }
                layer.temporal_decay.reset(self.lambda);
            }
        }
        Ok(())
    }

    pub fn maintenance_units(&self) -> Result<u64> {
        let mut total = self.network_units()?;
        let layer_units = self.layer_units()?;
        let geometric_units = self.kernel_units()?;
        let temporal_units = self.temporal_kernel_units()?;
        for layer in &self.layers {
            total = total
                .checked_add(layer_units)
                .and_then(|n| {
                    n.checked_add(
                        ((layer.sigma.len() + layer.cells.len()) as u64)
                            .checked_mul(geometric_units)?,
                    )
                })
                .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
            if self.mode.temporal() {
                total = total
                    .checked_add(
                        ((layer.temporal_sigma.len() + layer.temporal_cells.len()) as u64)
                            .checked_mul(temporal_units)
                            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?,
                    )
                    .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
            }
        }
        Ok(total)
    }

    pub fn cell_value(cell: &Kernel<S>) -> f64 {
        cell.norm2 / (cell.norm2 + cell.variance.to_f64())
    }

    pub fn step(&mut self, presentation: &[Vec<f64>], detailed_report: bool) -> Result<StepReport> {
        let presentation = self.presentation(presentation)?;
        let mut transformations = Vec::new();
        self.force_solvency(&mut transformations)?;
        let maintenance_open = self.maintenance_units()?;
        let layer_count_start = self.layers.len();
        let mut concept_readout = Vec::new();
        let mut readout_layers = 0usize;
        let mut sequence_readout = Vec::new();
        let mut prediction_readout = Vec::new();
        let mut all_seed_requests: Vec<(Space, usize, Kernel64)> = Vec::new();
        let mut layer_reports = Vec::new();
        let mut temporal_reports = Vec::new();
        let mut contexts: Vec<Option<Kernel64>> = vec![None; layer_count_start];
        let mut frontier_requested = false;

        // Complete geometric recursion first. Temporal cognition observes the
        // resulting contexts but can never feed back into geometry in the same step.
        let mut current = presentation;
        for (layer_index, context_slot) in contexts.iter_mut().enumerate().take(layer_count_start) {
            if current.is_empty() {
                break;
            }
            let result = self.process_layer(layer_index, &current, detailed_report)?;
            *context_slot = result.context.clone();
            if !result.readout.is_empty() {
                readout_layers += 1;
                concept_readout.extend(result.readout);
            }
            transformations.extend(result.transformations);
            all_seed_requests.extend(
                result
                    .seed_requests
                    .into_iter()
                    .map(|seed| (Space::Geometry, layer_index, seed)),
            );
            if let Some(report) = result.report {
                layer_reports.push(report);
            }
            if layer_index + 1 == layer_count_start && !result.output.is_empty() && self.beta > 0.0
            {
                frontier_requested = true;
            }
            current = result.output;
        }

        if self.mode.temporal() {
            for (layer_index, context_slot) in contexts.iter().enumerate().take(layer_count_start) {
                let previous = self.layers[layer_index]
                    .previous
                    .as_ref()
                    .map(|kernel| Kernel64 {
                        weight: kernel.weight(),
                        center: scalar_vec_to_f64(&kernel.center),
                        variance: kernel.variance.to_f64(),
                    });
                let context = context_slot.as_ref();

                // Prediction reads only temporal CELLs that already existed
                // before this layer's temporal learning phase. It projects
                // source/target centers and never reconstructs endpoint variance.
                if self.mode.predictive() {
                    if let Some(context) = context {
                        for cell in &self.layers[layer_index].temporal_cells {
                            debug_assert_eq!(cell.center.len(), self.dimension * 2);
                            let source = &cell.center[..self.dimension];
                            if point_concern_scalar(&context.center, source) {
                                prediction_readout.push(Prediction {
                                    universe: self.universe.clone(),
                                    current_context: context.center.clone(),
                                    recognised_source: scalar_vec_to_f64(source),
                                    predicted_successor: scalar_vec_to_f64(
                                        &cell.center[self.dimension..],
                                    ),
                                });
                            }
                        }
                    }
                }

                if let (Some(previous), Some(context)) = (previous.as_ref(), context) {
                    let atom = self.temporal_atom(previous, context);
                    let result = self.process_temporal(layer_index, &[atom], detailed_report)?;
                    transformations.extend(result.transformations);
                    all_seed_requests.extend(
                        result
                            .seed_requests
                            .into_iter()
                            .map(|seed| (Space::Temporal, layer_index, seed)),
                    );
                    for recognition in result.readout {
                        let input = recognition.local_input.as_ref();
                        let recognised = &recognition.recognised;
                        debug_assert_eq!(input.len(), self.dimension * 2);
                        debug_assert_eq!(recognised.len(), self.dimension * 2);
                        sequence_readout.push(TemporalRecognition {
                            universe: self.universe.clone(),
                            previous_input: Arc::from(input[..self.dimension].to_vec()),
                            current_input: Arc::from(input[self.dimension..].to_vec()),
                            previous_recognised: recognised[..self.dimension].to_vec(),
                            current_recognised: recognised[self.dimension..].to_vec(),
                        });
                    }
                    if let Some(report) = result.report {
                        temporal_reports.push(report);
                    }
                }

                // P_k is causal state, not learned memory: it advances even at eta=0.
                let next_previous = context_slot
                    .as_ref()
                    .map(|context| self.project_previous(context))
                    .transpose()?;
                self.layers[layer_index].previous = next_previous;
            }
        }

        // One material transaction spans both kinds of seed and the optional
        // new frontier layer. The economy never selects a subset. Admission
        // is evaluated on the geometry that will actually persist after
        // scalar projection: a raw seed can become zero, covered, or an exact
        // clone only at that boundary (notably in f32).
        let mut projected_geometry: Vec<Vec<Kernel<S>>> =
            (0..layer_count_start).map(|_| Vec::new()).collect();
        let mut projected_temporal: Vec<Vec<Kernel<S>>> =
            (0..layer_count_start).map(|_| Vec::new()).collect();
        for (space, layer_index, seed) in all_seed_requests {
            let projected = project_kernel::<S>(seed)?;
            if zero_scalar_vec(&projected.center) {
                continue;
            }
            let cells = match space {
                Space::Geometry => &self.layers[layer_index].cells,
                Space::Temporal => &self.layers[layer_index].temporal_cells,
            };
            if cells.iter().any(|cell| concern_kernel(cell, &projected).0) {
                continue;
            }
            match space {
                Space::Geometry => projected_geometry[layer_index].push(projected),
                Space::Temporal => projected_temporal[layer_index].push(projected),
            }
        }

        let mut future_geometry: Vec<Option<Vec<Kernel<S>>>> =
            (0..layer_count_start).map(|_| None).collect();
        let mut future_temporal: Vec<Option<Vec<Kernel<S>>>> =
            (0..layer_count_start).map(|_| None).collect();
        let mut geometric_seeds = 0usize;
        let mut temporal_seeds = 0usize;
        let mut net_new_geometry = 0usize;
        let mut net_new_temporal = 0usize;

        for layer_index in 0..layer_count_start {
            if !projected_geometry[layer_index].is_empty() {
                geometric_seeds = geometric_seeds
                    .checked_add(projected_geometry[layer_index].len())
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
                let existing_len = self.layers[layer_index].sigma.len();
                let mut future = self.layers[layer_index].sigma.clone();
                future.append(&mut projected_geometry[layer_index]);
                let future = coalesce_projected(future)?;
                net_new_geometry = net_new_geometry
                    .checked_add(future.len().saturating_sub(existing_len))
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
                future_geometry[layer_index] = Some(future);
            }
            if !projected_temporal[layer_index].is_empty() {
                temporal_seeds = temporal_seeds
                    .checked_add(projected_temporal[layer_index].len())
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
                let existing_len = self.layers[layer_index].temporal_sigma.len();
                let mut future = self.layers[layer_index].temporal_sigma.clone();
                future.append(&mut projected_temporal[layer_index]);
                let future = coalesce_projected(future)?;
                net_new_temporal = net_new_temporal
                    .checked_add(future.len().saturating_sub(existing_len))
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
                future_temporal[layer_index] = Some(future);
            }
        }

        let seed_count = geometric_seeds
            .checked_add(temporal_seeds)
            .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
        let mut growth_cost = if frontier_requested {
            self.layer_units()?
        } else {
            0
        };
        let net_new_geometry = u64::try_from(net_new_geometry)
            .map_err(|_| Error::Invalid("material accounting overflow".into()))?;
        let net_new_temporal = u64::try_from(net_new_temporal)
            .map_err(|_| Error::Invalid("material accounting overflow".into()))?;
        let geometry_growth = net_new_geometry
            .checked_mul(self.kernel_units()?)
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
        let temporal_growth = net_new_temporal
            .checked_mul(self.temporal_kernel_units()?)
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
        growth_cost = growth_cost
            .checked_add(geometry_growth)
            .and_then(|value| value.checked_add(temporal_growth))
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;

        let transaction_requested = seed_count > 0 || frontier_requested;
        if transaction_requested {
            let payable = self
                .maintenance_units()?
                .checked_add(growth_cost)
                .is_some_and(|n| n <= self.budget_units);
            if payable {
                for layer_index in 0..layer_count_start {
                    if let Some(future) = future_geometry[layer_index].take() {
                        self.layers[layer_index].sigma = future;
                    }
                    if let Some(future) = future_temporal[layer_index].take() {
                        self.layers[layer_index].temporal_sigma = future;
                    }
                }
                if frontier_requested {
                    self.layers.push(Layer {
                        sigma: Vec::new(),
                        cells: Vec::new(),
                        cell_decay: Arc::new(DecayClock::new(0, self.lambda)),
                        temporal_sigma: Vec::new(),
                        temporal_cells: Vec::new(),
                        temporal_decay: Arc::new(DecayClock::new(0, self.lambda)),
                        previous: None,
                    });
                }
                transformations.push(Transformation::GrowthCommit {
                    geometric_seeds,
                    temporal_seeds,
                    seeds: seed_count,
                    layer_created: frontier_requested,
                    units: growth_cost,
                });
            } else {
                transformations.push(Transformation::GrowthReject {
                    geometric_seeds,
                    temporal_seeds,
                    seeds: seed_count,
                    layer_requested: frontier_requested,
                    units: growth_cost,
                });
            }
        }

        self.steps_seen = self
            .steps_seen
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("steps_seen overflow".into()))?;
        let maintenance_end = self.maintenance_units()?;
        if maintenance_end > self.budget_units {
            return Err(Error::Inexecutable(
                "internal error: post-step state exceeds budget".into(),
            ));
        }

        if readout_layers <= 1 {
            dedup_single_layer_recognitions(&mut concept_readout);
        } else {
            sort_dedup_recognitions(&mut concept_readout);
        }
        sort_dedup_temporal_recognitions(&mut sequence_readout);
        sort_dedup_predictions(&mut prediction_readout);
        let readout = match self.mode {
            Mode::Geometry => Readout::Geometry(concept_readout),
            Mode::Temporal => Readout::Temporal {
                concepts: concept_readout,
                sequences: sequence_readout,
            },
            Mode::Predictive => Readout::Predictive {
                concepts: concept_readout,
                sequences: sequence_readout,
                predictions: prediction_readout,
            },
        };
        Ok(StepReport {
            step_index: self.steps_seen - 1,
            readout,
            transformations,
            maintenance_open_units: maintenance_open,
            maintenance_units: maintenance_end,
            budget_units: self.budget_units,
            layer_reports,
            temporal_reports,
        })
    }

    fn presentation(&self, value: &[Vec<f64>]) -> Result<Vec<Atom>> {
        if value.is_empty() {
            return Err(Error::Invalid(
                "external presentation must be a nonempty sequence of vectors".into(),
            ));
        }
        let total = value.len() as f64;
        let mut atoms = Vec::with_capacity(value.len());
        for (i, x) in value.iter().enumerate() {
            if x.len() != self.dimension {
                return Err(Error::Invalid(format!(
                    "presentation[{i}] must have dimension {}",
                    self.dimension
                )));
            }
            let mut canonical = Vec::with_capacity(self.dimension);
            let mut norm_sum = 0.0;
            let mut norm_correction = 0.0;
            let mut zero = true;
            for &component in x {
                if !component.is_finite() {
                    return Err(Error::Invalid(format!(
                        "presentation[{i}] must contain only finite reals"
                    )));
                }
                let component = structural_zero(component);
                zero &= component == 0.0;
                compensated_add(&mut norm_sum, &mut norm_correction, component * component);
                canonical.push(component);
            }
            atoms.push(Atom {
                x: Arc::from(canonical),
                // Count exact duplicate geometry first. The final uniform
                // mass is count / n, avoiding split-dependent accumulation
                // of rounded 1/n atoms.
                r: 1.0,
                variance: 0.0,
                norm2: compensated_finish(norm_sum, norm_correction),
                zero,
            });
        }
        let mut atoms = coalesce_atoms(atoms);
        for atom in &mut atoms {
            atom.r /= total;
        }
        Ok(atoms)
    }

    fn process_compartment(
        &mut self,
        layer_index: usize,
        presentation: &[Atom],
        detailed: bool,
        space: Space,
        dimension: usize,
    ) -> Result<LayerResult> {
        let mut targets = mem::take(&mut self.scratch_targets);
        let mut concerned = mem::take(&mut self.scratch_concerned);
        let mut unknown = mem::take(&mut self.scratch_unknown);
        concerned.clear();
        unknown.clear();
        targets.changed.clear();

        let layer = &mut self.layers[layer_index];
        let mut cell_epoch = layer.cell_decay.epoch();
        if cell_epoch == u32::MAX {
            let lambda = layer.cell_decay.lambda();
            for cell in &mut layer.cells {
                cell.materialize_weight_at(cell_epoch, lambda);
                cell.decay_epoch = 0;
            }
            layer.cell_decay.epoch.store(0, AtomicOrdering::Relaxed);
            cell_epoch = 0;
        }
        let cell_decay_clock = layer.cell_decay.clone();
        let mut old_cells = mem::take(&mut layer.cells);
        let old_sigma = mem::take(&mut layer.sigma);
        debug_assert!(old_cells.iter().all(|kernel| !kernel.dirty));
        debug_assert!(old_sigma.iter().all(|kernel| !kernel.dirty));
        let cell_count_before = old_cells.len();
        let sigma_count_before = old_sigma.len();

        // External singletons keep the specialised V=0 path. Internal
        // contextual atoms have V>0 and use the general total-variance path.
        let point_single = presentation.len() == 1 && presentation[0].variance == 0.0;
        if self.beta > 0.0 {
            targets.reset(old_cells.len(), dimension, !point_single);
        }
        let mut cell_received = detailed.then(|| vec![0.0; old_cells.len()]);
        let mut readout = Vec::new();
        let mut context_contributions: Vec<(usize, f64)> = Vec::new();
        let mut recognised_atoms = 0usize;

        // CONCERN -> ALLOCATE -> RECOGNISE from the frozen CELL state.
        // The vertical context is built only from exact recognised snapshot
        // values. Learning responsibilities never weight that geometry.
        for (atom_index, atom) in presentation.iter().enumerate() {
            if atom.zero {
                unknown.push(atom_index);
                continue;
            }
            let d0 = atom.norm2;
            concerned.clear();
            let mut max_w = 0.0f64;
            let candidate_range = first_coordinate_candidate_range(&old_cells, atom.x[0], d0);
            for ci in candidate_range {
                let (ok, gain, distance2) =
                    concern_scalar(&old_cells[ci], &atom.x, atom.variance, d0);
                if ok {
                    let weight = old_cells[ci].materialize_weight_at(cell_epoch, self.lambda);
                    concerned.push((ci, gain));
                    if point_single && self.beta > 0.0 {
                        targets.mark_single(ci, distance2);
                    }
                    max_w = max_w.max(weight);
                }
            }
            if concerned.is_empty() {
                unknown.push(atom_index);
                continue;
            }

            recognised_atoms += 1;
            for (ci, score) in &mut concerned {
                *score *= old_cells[*ci].weight.to_f64() / max_w;
            }
            let denominator = stable_sum(concerned.iter().map(|&(_, score)| score));
            for &(ci, score) in &concerned {
                let cell = &old_cells[ci];
                let rho = atom.r * score / denominator;
                if self.beta > 0.0 {
                    if point_single {
                        targets.set_single_weight(ci, rho);
                    } else {
                        targets.add_atom(ci, atom_index, rho);
                    }
                }
                if let Some(received) = &mut cell_received {
                    received[ci] += rho;
                }
                readout.push(Recognition {
                    universe: self.universe.clone(),
                    local_input: atom.x.clone(),
                    recognised: scalar_vec_to_f64(&cell.center),
                });
            }

            // R_s is the exact quotient of recognised snapshot values by
            // center only (CELL dispersion/identity do not create concepts).
            // Each distinct recognised value gets r_s / |R_s|; the resulting
            // contextual mass is therefore exactly the recognised input mass.
            let mut unique_count = 0usize;
            let mut previous: Option<&[S]> = None;
            for &(ci, _) in &concerned {
                let center = old_cells[ci].center.as_slice();
                if previous.is_none_or(|prev| prev != center) {
                    unique_count += 1;
                    previous = Some(center);
                }
            }
            debug_assert!(unique_count > 0);
            let share = atom.r / unique_count as f64;
            previous = None;
            for &(ci, _) in &concerned {
                let center = old_cells[ci].center.as_slice();
                if previous.is_some_and(|prev| prev == center) {
                    continue;
                }
                context_contributions.push((ci, share));
                previous = Some(center);
            }
        }

        let context = build_context_kernel(&old_cells, &context_contributions, dimension);

        // Context is frozen from L^- before local learning. A singleton
        // context (V=0) has no vertical authority; neither does an exactly
        // zero-centered context, because Auxein has no canonical direction
        // for such a symmetric relation.
        let context_emitted = context
            .as_ref()
            .is_some_and(|kernel| kernel.variance > 0.0 && !zero_f64_vec(&kernel.center));
        let output = if context_emitted {
            let kernel = context.as_ref().expect("emitted context exists");
            vec![Atom {
                x: Arc::from(kernel.center.clone()),
                r: kernel.weight,
                variance: kernel.variance,
                norm2: norm2(&kernel.center),
                zero: false,
            }]
        } else {
            Vec::new()
        };

        if self.beta == 0.0 {
            let unknown_atom_count = unknown.len();
            layer.cells = old_cells;
            layer.sigma = old_sigma;
            self.scratch_targets = targets;
            self.scratch_concerned = concerned;
            self.scratch_unknown = unknown;
            let report = if detailed {
                Some(LayerReport {
                    phase: space,
                    layer_index,
                    input_atom_count: presentation.len(),
                    input_mass: stable_sum(presentation.iter().map(|a| a.r)),
                    unknown_atom_count,
                    recognised_atom_count: recognised_atoms,
                    cell_count_before,
                    cell_count_after: layer.cells.len(),
                    sigma_count_before,
                    sigma_count_after: layer.sigma.len(),
                    promoted: 0,
                    seed_requests: 0,
                    context_emitted,
                    output_atom_count: output.len(),
                    output_mass: output.first().map_or(0.0, |a| a.r),
                    context_center: context.as_ref().map(|k| k.center.clone()),
                    context_variance: context.as_ref().map(|k| k.variance),
                    recognition_count: readout.len(),
                    cell_responsibility_mass: cell_received.unwrap_or_default(),
                })
            } else {
                None
            };
            return Ok(LayerResult {
                output,
                context,
                readout,
                seed_requests: Vec::new(),
                transformations: Vec::new(),
                report,
            });
        }

        let mut current_cells = old_cells;
        let next_cell_epoch = cell_epoch
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("cell decay epoch overflow".into()))?;
        cell_decay_clock
            .epoch
            .store(next_cell_epoch, AtomicOrdering::Relaxed);
        if point_single {
            targets.apply_single_cell_population(
                &mut current_cells,
                &presentation[0].x,
                self.beta,
                self.lambda,
                next_cell_epoch,
            )?;
        } else {
            targets.apply_cell_population(
                &mut current_cells,
                presentation,
                self.beta,
                self.lambda,
                next_cell_epoch,
            )?;
        }
        let cell_indices_stable = retain_touched_nonzero(&mut current_cells, &targets.touched);
        let (mut current_cells, cell_order_stable) =
            coalesce_touched(current_cells, &targets.touched, cell_indices_stable)?;
        if cell_order_stable {
            targets.changed.extend(
                targets
                    .touched
                    .iter()
                    .copied()
                    .filter(|&index| current_cells[index].dirty),
            );
        } else {
            targets.changed.extend(
                current_cells
                    .iter()
                    .enumerate()
                    .filter_map(|(index, cell)| cell.dirty.then_some(index)),
            );
        }

        targets.reset(old_sigma.len(), dimension, !point_single);
        let mut seed_requests = Vec::new();
        for &atom_index in &unknown {
            let atom = &presentation[atom_index];
            if atom.zero {
                continue;
            }
            let d0 = atom.norm2;
            concerned.clear();
            let mut max_w = 0.0f64;
            let candidate_range = first_coordinate_candidate_range(&old_sigma, atom.x[0], d0);
            for si in candidate_range {
                let sigma = &old_sigma[si];
                let (ok, gain, distance2) = concern_scalar(sigma, &atom.x, atom.variance, d0);
                if ok {
                    concerned.push((si, gain));
                    if point_single {
                        targets.mark_single(si, distance2);
                    }
                    max_w = max_w.max(sigma.weight.to_f64());
                }
            }
            if !concerned.is_empty() {
                for (si, score) in &mut concerned {
                    *score *= old_sigma[*si].weight.to_f64() / max_w;
                }
                let denominator = stable_sum(concerned.iter().map(|&(_, score)| score));
                for &(si, score) in &concerned {
                    let tau = atom.r * score / denominator;
                    if point_single {
                        targets.set_single_weight(si, tau);
                    } else {
                        targets.add_atom(si, atom_index, tau);
                    }
                }
            } else {
                seed_requests.push(Kernel64 {
                    weight: self.beta * atom.r,
                    center: atom.x.to_vec(),
                    variance: atom.variance,
                });
            }
        }

        let mut updated_sigma = old_sigma;
        if point_single {
            targets.apply_single_population(
                &mut updated_sigma,
                &presentation[0].x,
                self.beta,
                self.lambda,
            )?;
        } else {
            targets.apply_population(&mut updated_sigma, presentation, self.beta, self.lambda)?;
        }
        let sigma_indices_stable = retain_touched_nonzero(&mut updated_sigma, &targets.touched);
        let (updated_sigma, _) =
            coalesce_touched(updated_sigma, &targets.touched, sigma_indices_stable)?;

        let mut transformations = Vec::new();
        let mut promoted = Vec::new();
        let mut remaining_sigma = Vec::new();
        for kernel in updated_sigma {
            if kernel.weight.to_f64() > self.beta && !zero_scalar_vec(&kernel.center) {
                promoted.push(kernel);
            } else {
                remaining_sigma.push(kernel);
            }
        }
        let promoted_count = promoted.len();
        if promoted_count > 0 {
            current_cells.extend(promoted.into_iter().map(|mut kernel| {
                kernel.dirty = true;
                kernel.bind_decay_clock(cell_decay_clock.clone(), next_cell_epoch);
                kernel
            }));
            current_cells = coalesce_dirty(current_cells)?;
            targets.changed.clear();
            targets.changed.extend(
                current_cells
                    .iter()
                    .enumerate()
                    .filter_map(|(index, cell)| cell.dirty.then_some(index)),
            );
            transformations.push(Transformation::Promote {
                space,
                layer: layer_index,
                count: promoted_count,
            });
        }

        let mut admissible_seeds = Vec::new();
        if remaining_sigma.is_empty() && seed_requests.is_empty() {
            layer.sigma.clear();
        } else {
            let changed_indices = &targets.changed;

            let mut cleaned_sigma = Vec::new();
            for mut sigma in remaining_sigma {
                let candidate_range = first_coordinate_candidate_range(
                    &current_cells,
                    sigma.center[0].to_f64(),
                    sigma.norm2,
                );
                let covered = if sigma.dirty {
                    current_cells[candidate_range.clone()]
                        .iter()
                        .any(|cell| concern_kernel(cell, &sigma).0)
                } else {
                    let first = changed_indices.partition_point(|&ci| ci < candidate_range.start);
                    let last = changed_indices.partition_point(|&ci| ci < candidate_range.end);
                    changed_indices[first..last]
                        .iter()
                        .any(|&ci| concern_kernel(&current_cells[ci], &sigma).0)
                };
                if !covered {
                    sigma.dirty = false;
                    cleaned_sigma.push(sigma);
                }
            }
            layer.sigma = cleaned_sigma;

            for seed in coalesce_kernel64(seed_requests) {
                let seed_norm2 = norm2(&seed.center);
                let candidate_range =
                    first_coordinate_candidate_range(&current_cells, seed.center[0], seed_norm2);
                let first = changed_indices.partition_point(|&ci| ci < candidate_range.start);
                let last = changed_indices.partition_point(|&ci| ci < candidate_range.end);
                let covered = changed_indices[first..last].iter().any(|&ci| {
                    concern_raw_kernel(&current_cells[ci], &seed.center, seed.variance, seed_norm2)
                        .0
                });
                if !covered {
                    admissible_seeds.push(seed);
                }
            }
        }
        for &index in &targets.changed {
            current_cells[index].dirty = false;
        }
        layer.cells = current_cells;
        let unknown_atom_count = unknown.len();
        self.scratch_targets = targets;
        self.scratch_concerned = concerned;
        self.scratch_unknown = unknown;

        let report = if detailed {
            Some(LayerReport {
                phase: space,
                layer_index,
                input_atom_count: presentation.len(),
                input_mass: stable_sum(presentation.iter().map(|a| a.r)),
                unknown_atom_count,
                recognised_atom_count: recognised_atoms,
                cell_count_before,
                cell_count_after: layer.cells.len(),
                sigma_count_before,
                sigma_count_after: layer.sigma.len(),
                promoted: promoted_count,
                seed_requests: admissible_seeds.len(),
                context_emitted,
                output_atom_count: output.len(),
                output_mass: output.first().map_or(0.0, |a| a.r),
                context_center: context.as_ref().map(|k| k.center.clone()),
                context_variance: context.as_ref().map(|k| k.variance),
                recognition_count: readout.len(),
                cell_responsibility_mass: cell_received.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(LayerResult {
            output,
            context,
            readout,
            seed_requests: admissible_seeds,
            transformations,
            report,
        })
    }

    fn process_layer(
        &mut self,
        layer_index: usize,
        presentation: &[Atom],
        detailed: bool,
    ) -> Result<LayerResult> {
        self.process_compartment(
            layer_index,
            presentation,
            detailed,
            Space::Geometry,
            self.dimension,
        )
    }

    fn process_temporal(
        &mut self,
        layer_index: usize,
        presentation: &[Atom],
        detailed: bool,
    ) -> Result<LayerResult> {
        {
            let layer = &mut self.layers[layer_index];
            mem::swap(&mut layer.sigma, &mut layer.temporal_sigma);
            mem::swap(&mut layer.cells, &mut layer.temporal_cells);
            mem::swap(&mut layer.cell_decay, &mut layer.temporal_decay);
        }
        let result = self.process_compartment(
            layer_index,
            presentation,
            detailed,
            Space::Temporal,
            self.dimension * 2,
        );
        {
            let layer = &mut self.layers[layer_index];
            mem::swap(&mut layer.sigma, &mut layer.temporal_sigma);
            mem::swap(&mut layer.cells, &mut layer.temporal_cells);
            mem::swap(&mut layer.cell_decay, &mut layer.temporal_decay);
        }
        let mut result = result?;
        result.output.clear();
        result.context = None;
        if let Some(report) = &mut result.report {
            report.context_emitted = false;
            report.output_atom_count = 0;
            report.output_mass = 0.0;
            report.context_center = None;
            report.context_variance = None;
        }
        Ok(result)
    }

    fn temporal_atom(&self, previous: &Kernel64, current: &Kernel64) -> Atom {
        let mut center = Vec::with_capacity(self.dimension * 2);
        center.extend_from_slice(&previous.center);
        center.extend_from_slice(&current.center);
        Atom {
            x: Arc::from(center.clone()),
            r: previous.weight * current.weight,
            variance: previous.variance + current.variance,
            norm2: norm2(&center),
            zero: zero_f64_vec(&center),
        }
    }

    fn project_previous(&self, context: &Kernel64) -> Result<Kernel<S>> {
        project_kernel::<S>(context.clone())
    }

    fn invalidate_previous(&mut self) {
        if self.mode.temporal() {
            for layer in &mut self.layers {
                layer.previous = None;
            }
        }
    }

    fn layer_has_cells(&self, layer: &Layer<S>) -> bool {
        !layer.cells.is_empty() || (self.mode.temporal() && !layer.temporal_cells.is_empty())
    }

    fn force_solvency(&mut self, transformations: &mut Vec<Transformation>) -> Result<()> {
        if self.maintenance_units()? <= self.budget_units {
            return Ok(());
        }

        // Work in progress is discarded simultaneously in both spaces.
        let removed_sigma: usize = self
            .layers
            .iter()
            .map(|l| {
                l.sigma.len()
                    + if self.mode.temporal() {
                        l.temporal_sigma.len()
                    } else {
                        0
                    }
            })
            .sum();
        if removed_sigma > 0 {
            for layer in &mut self.layers {
                layer.sigma.clear();
                if self.mode.temporal() {
                    layer.temporal_sigma.clear();
                }
            }
            transformations.push(Transformation::ClearSigma {
                count: removed_sigma,
            });
        }

        let mut trimmed = 0;
        while self.layers.len() > 1
            && self
                .layers
                .last()
                .is_some_and(|layer| !self.layer_has_cells(layer))
        {
            self.layers.pop();
            trimmed += 1;
        }
        if trimmed > 0 {
            transformations.push(Transformation::TrimLayers { count: trimmed });
        }
        if self.maintenance_units()? <= self.budget_units {
            self.invalidate_previous();
            return Ok(());
        }

        // One K ordering spans geometric and temporal knowledge. Equal K is
        // destroyed as a whole wave independently of its compartment.
        let mut valued: Vec<(f64, usize, Space)> = Vec::new();
        let mut geometric_counts: Vec<usize> = self.layers.iter().map(|l| l.cells.len()).collect();
        let mut temporal_counts: Vec<usize> = self
            .layers
            .iter()
            .map(|l| {
                if self.mode.temporal() {
                    l.temporal_cells.len()
                } else {
                    0
                }
            })
            .collect();
        for (li, layer) in self.layers.iter().enumerate() {
            valued.extend(
                layer
                    .cells
                    .iter()
                    .map(|cell| (Self::cell_value(cell), li, Space::Geometry)),
            );
            if self.mode.temporal() {
                valued.extend(
                    layer
                        .temporal_cells
                        .iter()
                        .map(|cell| (Self::cell_value(cell), li, Space::Temporal)),
                );
            }
        }
        if valued.is_empty() {
            self.invalidate_previous();
            if self.maintenance_units()? > self.budget_units {
                return Err(Error::Inexecutable(
                    "minimal Auxein state exceeds the execution budget".into(),
                ));
            }
            return Ok(());
        }
        valued.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut active_layers = self.layers.len();
        let mut cutoff = None;
        let mut removed_cells = 0usize;
        let mut waves = 0usize;
        let mut position = 0usize;
        while position < valued.len() {
            let k = valued[position].0;
            let mut stop = position;
            while stop < valued.len() && valued[stop].0 == k {
                let (_, li, space) = valued[stop];
                match space {
                    Space::Geometry => geometric_counts[li] -= 1,
                    Space::Temporal => temporal_counts[li] -= 1,
                }
                removed_cells += 1;
                stop += 1;
            }
            waves += 1;
            while active_layers > 1
                && geometric_counts[active_layers - 1] == 0
                && temporal_counts[active_layers - 1] == 0
            {
                active_layers -= 1;
            }
            let geometric_total: usize = geometric_counts[..active_layers].iter().sum();
            let temporal_total: usize = temporal_counts[..active_layers].iter().sum();
            let simulated = self
                .network_units()?
                .checked_add(
                    self.layer_units()?
                        .checked_mul(active_layers as u64)
                        .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?,
                )
                .and_then(|n| {
                    n.checked_add((geometric_total as u64).checked_mul(self.kernel_units().ok()?)?)
                })
                .and_then(|n| {
                    n.checked_add(
                        (temporal_total as u64).checked_mul(self.temporal_kernel_units().ok()?)?,
                    )
                })
                .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
            cutoff = Some(k);
            if simulated <= self.budget_units {
                break;
            }
            position = stop;
        }
        let cutoff = cutoff.unwrap_or(f64::INFINITY);

        for layer in &mut self.layers {
            layer.cells.retain(|cell| Self::cell_value(cell) > cutoff);
            if self.mode.temporal() {
                layer
                    .temporal_cells
                    .retain(|cell| Self::cell_value(cell) > cutoff);
            }
        }
        let mut trimmed_after = 0;
        while self.layers.len() > 1
            && self
                .layers
                .last()
                .is_some_and(|layer| !self.layer_has_cells(layer))
        {
            self.layers.pop();
            trimmed_after += 1;
        }
        transformations.push(Transformation::DestroyCells {
            count: removed_cells,
            waves,
            k_through: cutoff,
        });
        if trimmed_after > 0 {
            transformations.push(Transformation::TrimLayers {
                count: trimmed_after,
            });
        }
        self.invalidate_previous();
        if self.maintenance_units()? > self.budget_units {
            return Err(Error::Inexecutable(
                "forced solvency did not reach the budget".into(),
            ));
        }
        Ok(())
    }

    pub fn summary(&self) -> Result<Summary> {
        let maintenance = self.maintenance_units()?;
        Ok(Summary {
            steps_seen: self.steps_seen,
            dimension: self.dimension,
            universe: self.universe.to_string(),
            scalar: S::NAME,
            memory: self.memory.to_f64(),
            eta: self.eta.to_f64(),
            mode: self.mode,
            chi: self.chi,
            alpha: self.alpha,
            effective_alpha: self.beta,
            layer_count: self.layers.len(),
            cells_per_layer: self.layers.iter().map(|l| l.cells.len()).collect(),
            sigma_per_layer: self.layers.iter().map(|l| l.sigma.len()).collect(),
            temporal_cells_per_layer: self
                .layers
                .iter()
                .map(|l| {
                    if self.mode.temporal() {
                        l.temporal_cells.len()
                    } else {
                        0
                    }
                })
                .collect(),
            temporal_sigma_per_layer: self
                .layers
                .iter()
                .map(|l| {
                    if self.mode.temporal() {
                        l.temporal_sigma.len()
                    } else {
                        0
                    }
                })
                .collect(),
            previous_context_per_layer: self
                .layers
                .iter()
                .map(|l| self.mode.temporal() && l.previous.is_some())
                .collect(),
            maintenance_units: maintenance,
            budget: self.budget_equivalent()?,
            budget_units: self.budget_units,
            budget_margin_units: self.budget_units as i128 - maintenance as i128,
            is_solvent: maintenance <= self.budget_units,
        })
    }

    pub fn export_json(&self) -> String {
        let mut out = String::with_capacity(256 + self.maintenance_units().unwrap_or(0) as usize);
        out.push('{');
        out.push_str("\"format_version\":4,");
        out.push_str("\"dimension\":");
        out.push_str(&self.dimension.to_string());
        out.push_str(",\"scalar\":");
        json::quote(&mut out, S::NAME);
        out.push_str(",\"memory\":");
        push_float(&mut out, self.memory.to_f64());
        out.push_str(",\"eta\":");
        push_float(&mut out, self.eta.to_f64());
        out.push_str(",\"mode\":");
        json::quote(&mut out, self.mode.as_str());
        out.push_str(",\"steps_seen\":");
        out.push_str(&self.steps_seen.to_string());
        out.push_str(",\"layers\":[");
        for (li, layer) in self.layers.iter().enumerate() {
            if li > 0 {
                out.push(',');
            }
            out.push_str("{\"sigma\":[");
            write_kernel_list(&mut out, &layer.sigma);
            out.push_str("],\"cells\":[");
            write_kernel_list(&mut out, &layer.cells);
            if self.mode.temporal() {
                out.push_str("],\"temporal_sigma\":[");
                write_kernel_list(&mut out, &layer.temporal_sigma);
                out.push_str("],\"temporal_cells\":[");
                write_kernel_list(&mut out, &layer.temporal_cells);
                out.push_str("],\"previous\":");
                if let Some(previous) = &layer.previous {
                    write_kernel(&mut out, previous);
                } else {
                    out.push_str("null");
                }
                out.push('}');
            } else {
                out.push_str("]}");
            }
        }
        out.push_str("]}");
        out
    }

    pub fn from_json(json_text: &str, budget: Budget, universe: impl Into<String>) -> Result<Self> {
        let parsed = json::parse(json_text)?;
        let state = ParsedState::from_json(parsed)?;
        if state.scalar != S::NAME {
            return Err(Error::Invalid(format!(
                "state scalar is {}, requested engine is {}",
                state.scalar,
                S::NAME
            )));
        }
        Self::from_parsed_state(state, budget, universe.into())
    }

    fn from_parsed_state(state: ParsedState, budget: Budget, universe: String) -> Result<Self> {
        let mut network = Self::new_with_mode(
            state.dimension,
            state.memory,
            state.eta,
            state.mode,
            budget,
            universe,
        )?;
        if network.memory.to_f64() != state.memory || network.eta.to_f64() != state.eta {
            return Err(Error::Invalid(
                "state.memory/state.eta are not exactly representable in state.scalar".into(),
            ));
        }
        if state.layers.is_empty() {
            return Err(Error::Invalid(
                "state.layers must be a nonempty list".into(),
            ));
        }
        let mut layers = Vec::with_capacity(state.layers.len());
        for (li, layer) in state.layers.into_iter().enumerate() {
            let sigma = load_kernel_list::<S>(
                layer.sigma,
                state.dimension,
                &format!("layers[{li}].sigma"),
            )?;
            let cells = load_kernel_list::<S>(
                layer.cells,
                state.dimension,
                &format!("layers[{li}].cells"),
            )?;
            let clock = Arc::new(DecayClock::new(0, network.lambda));
            let mut cells = cells;
            for cell in &mut cells {
                cell.bind_decay_clock(clock.clone(), 0);
            }

            let temporal_sigma = load_kernel_list::<S>(
                layer.temporal_sigma,
                2 * state.dimension,
                &format!("layers[{li}].temporal_sigma"),
            )?;
            let temporal_cells = load_kernel_list::<S>(
                layer.temporal_cells,
                2 * state.dimension,
                &format!("layers[{li}].temporal_cells"),
            )?;
            let temporal_clock = Arc::new(DecayClock::new(0, network.lambda));
            let mut temporal_cells = temporal_cells;
            for cell in &mut temporal_cells {
                cell.bind_decay_clock(temporal_clock.clone(), 0);
            }
            let previous = load_optional_kernel::<S>(
                layer.previous,
                state.dimension,
                &format!("layers[{li}].previous"),
            )?;
            layers.push(Layer {
                sigma,
                cells,
                cell_decay: clock,
                temporal_sigma,
                temporal_cells,
                temporal_decay: temporal_clock,
                previous,
            });
        }
        network.layers = layers;
        network.steps_seen = state.steps_seen;
        network.validate_state()?;
        Ok(network)
    }

    fn validate_state(&mut self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(Error::Invalid("network must contain L0".into()));
        }
        for (li, layer) in self.layers.iter_mut().enumerate() {
            sort_kernels(&mut layer.cells);
            sort_kernels(&mut layer.sigma);
            assert_unique_geometry(&layer.cells, &format!("layers[{li}].cells"))?;
            assert_unique_geometry(&layer.sigma, &format!("layers[{li}].sigma"))?;
            if layer.cells.iter().any(|cell| zero_scalar_vec(&cell.center))
                || layer
                    .sigma
                    .iter()
                    .any(|kernel| zero_scalar_vec(&kernel.center))
            {
                return Err(Error::Invalid(
                    "persistent geometric CELL/Sigma center must be nonzero".into(),
                ));
            }
            if self.mode.temporal() {
                sort_kernels(&mut layer.temporal_cells);
                sort_kernels(&mut layer.temporal_sigma);
                assert_unique_geometry(
                    &layer.temporal_cells,
                    &format!("layers[{li}].temporal_cells"),
                )?;
                assert_unique_geometry(
                    &layer.temporal_sigma,
                    &format!("layers[{li}].temporal_sigma"),
                )?;
                if layer
                    .temporal_cells
                    .iter()
                    .chain(layer.temporal_sigma.iter())
                    .any(|kernel| zero_scalar_vec(&kernel.center))
                {
                    return Err(Error::Invalid(
                        "persistent temporal CELL/Sigma center must be nonzero".into(),
                    ));
                }
                if layer
                    .previous
                    .as_ref()
                    .is_some_and(|kernel| kernel.center.len() != self.dimension)
                {
                    return Err(Error::Invalid(
                        "previous context has invalid dimension".into(),
                    ));
                }
            } else if !layer.temporal_cells.is_empty()
                || !layer.temporal_sigma.is_empty()
                || layer.previous.is_some()
            {
                return Err(Error::Invalid(
                    "geometry mode cannot contain temporal state".into(),
                ));
            }
        }
        if self.layers.len() > 1 {
            let n = self.layers.len();
            let last = &self.layers[n - 1];
            let previous = &self.layers[n - 2];
            if last.cells.is_empty()
                && last.sigma.is_empty()
                && last.temporal_cells.is_empty()
                && last.temporal_sigma.is_empty()
                && previous.cells.is_empty()
            {
                return Err(Error::Invalid(
                    "state contains redundant terminal layers".into(),
                ));
            }
        }
        Ok(())
    }
}

impl Network {
    pub fn new(
        scalar: &str,
        dimension: usize,
        memory: f64,
        eta: f64,
        budget: Budget,
        universe: impl Into<String>,
    ) -> Result<Self> {
        Self::new_with_mode(
            scalar,
            dimension,
            memory,
            eta,
            Mode::Geometry,
            budget,
            universe,
        )
    }

    pub fn new_with_mode(
        scalar: &str,
        dimension: usize,
        memory: f64,
        eta: f64,
        mode: Mode,
        budget: Budget,
        universe: impl Into<String>,
    ) -> Result<Self> {
        let universe = universe.into();
        match scalar {
            "f32" => Ok(Self::F32(Auxein::new_with_mode(
                dimension, memory, eta, mode, budget, universe,
            )?)),
            "f64" => Ok(Self::F64(Auxein::new_with_mode(
                dimension, memory, eta, mode, budget, universe,
            )?)),
            _ => Err(Error::Invalid("scalar must be 'f32' or 'f64'".into())),
        }
    }

    pub fn from_json(json_text: &str, budget: Budget, universe: impl Into<String>) -> Result<Self> {
        let parsed = json::parse(json_text)?;
        let state = ParsedState::from_json(parsed)?;
        let universe = universe.into();
        match state.scalar.as_str() {
            "f32" => Ok(Self::F32(Auxein::<f32>::from_parsed_state(
                state, budget, universe,
            )?)),
            "f64" => Ok(Self::F64(Auxein::<f64>::from_parsed_state(
                state, budget, universe,
            )?)),
            _ => Err(Error::Invalid("state.scalar must be 'f32' or 'f64'".into())),
        }
    }

    pub fn step(&mut self, presentation: &[Vec<f64>], detailed: bool) -> Result<StepReport> {
        match self {
            Self::F32(n) => n.step(presentation, detailed),
            Self::F64(n) => n.step(presentation, detailed),
        }
    }

    pub fn set_eta(&mut self, eta: f64) -> Result<()> {
        match self {
            Self::F32(n) => n.set_eta(eta),
            Self::F64(n) => n.set_eta(eta),
        }
    }

    pub fn set_budget(&mut self, budget: Budget) -> Result<()> {
        match self {
            Self::F32(n) => n.set_budget(budget),
            Self::F64(n) => n.set_budget(budget),
        }
    }

    pub fn summary(&self) -> Result<Summary> {
        match self {
            Self::F32(n) => n.summary(),
            Self::F64(n) => n.summary(),
        }
    }

    pub fn export_json(&self) -> String {
        match self {
            Self::F32(n) => n.export_json(),
            Self::F64(n) => n.export_json(),
        }
    }
}

#[derive(Debug)]
struct ParsedKernel {
    weight: f64,
    center: Vec<f64>,
    variance: f64,
}

#[derive(Debug)]
struct ParsedLayer {
    sigma: Vec<ParsedKernel>,
    cells: Vec<ParsedKernel>,
    temporal_sigma: Vec<ParsedKernel>,
    temporal_cells: Vec<ParsedKernel>,
    previous: Option<ParsedKernel>,
}

#[derive(Debug)]
struct ParsedState {
    dimension: usize,
    scalar: String,
    memory: f64,
    eta: f64,
    mode: Mode,
    steps_seen: u64,
    layers: Vec<ParsedLayer>,
}

impl ParsedState {
    fn from_json(value: json::Value) -> Result<Self> {
        let mut obj = expect_object(value, "state")?;
        exact_keys(
            &obj,
            &[
                "format_version",
                "dimension",
                "scalar",
                "memory",
                "eta",
                "mode",
                "steps_seen",
                "layers",
            ],
            "state",
        )?;
        let version = take_u64(&mut obj, "format_version", "state.format_version")?;
        if version != FORMAT_VERSION {
            return Err(Error::Invalid("unsupported format_version".into()));
        }
        let dimension_u64 = take_u64(&mut obj, "dimension", "state.dimension")?;
        let dimension = usize::try_from(dimension_u64)
            .map_err(|_| Error::Invalid("state.dimension is too large".into()))?;
        if dimension == 0 {
            return Err(Error::Invalid("invalid state.dimension".into()));
        }
        let scalar = take_string(&mut obj, "scalar", "state.scalar")?;
        if scalar != "f32" && scalar != "f64" {
            return Err(Error::Invalid("invalid state.scalar".into()));
        }
        let memory = take_f64(&mut obj, "memory", "state.memory")?;
        if memory <= 0.0 {
            return Err(Error::Invalid("state.memory must be positive".into()));
        }
        let eta = take_f64(&mut obj, "eta", "state.eta")?;
        if !(0.0..=1.0).contains(&eta) {
            return Err(Error::Invalid("state.eta must lie in [0,1]".into()));
        }
        let mode = Mode::parse(&take_string(&mut obj, "mode", "state.mode")?)?;
        let steps_seen = take_u64(&mut obj, "steps_seen", "state.steps_seen")?;
        let raw_layers = take_array(&mut obj, "layers", "state.layers")?;
        if raw_layers.is_empty() {
            return Err(Error::Invalid("state.layers must be nonempty".into()));
        }
        let mut layers = Vec::with_capacity(raw_layers.len());
        for (li, raw) in raw_layers.into_iter().enumerate() {
            let mut layer = expect_object(raw, &format!("state.layers[{li}]"))?;
            match mode {
                Mode::Geometry => {
                    exact_keys(&layer, &["sigma", "cells"], &format!("state.layers[{li}]"))?
                }
                Mode::Temporal | Mode::Predictive => exact_keys(
                    &layer,
                    &[
                        "sigma",
                        "cells",
                        "temporal_sigma",
                        "temporal_cells",
                        "previous",
                    ],
                    &format!("state.layers[{li}]"),
                )?,
            }
            let sigma = parse_kernel_list(
                take_array(&mut layer, "sigma", "sigma")?,
                dimension,
                &format!("state.layers[{li}].sigma"),
            )?;
            let cells = parse_kernel_list(
                take_array(&mut layer, "cells", "cells")?,
                dimension,
                &format!("state.layers[{li}].cells"),
            )?;
            let (temporal_sigma, temporal_cells, previous) = match mode {
                Mode::Geometry => (Vec::new(), Vec::new(), None),
                Mode::Temporal | Mode::Predictive => {
                    let temporal_sigma = parse_kernel_list(
                        take_array(&mut layer, "temporal_sigma", "temporal_sigma")?,
                        2 * dimension,
                        &format!("state.layers[{li}].temporal_sigma"),
                    )?;
                    let temporal_cells = parse_kernel_list(
                        take_array(&mut layer, "temporal_cells", "temporal_cells")?,
                        2 * dimension,
                        &format!("state.layers[{li}].temporal_cells"),
                    )?;
                    let previous_value = layer
                        .remove("previous")
                        .ok_or_else(|| Error::Invalid("missing state.layers[].previous".into()))?;
                    let previous = match previous_value {
                        json::Value::Null => None,
                        value => Some(parse_kernel_value(
                            value,
                            dimension,
                            &format!("state.layers[{li}].previous"),
                        )?),
                    };
                    (temporal_sigma, temporal_cells, previous)
                }
            };
            layers.push(ParsedLayer {
                sigma,
                cells,
                temporal_sigma,
                temporal_cells,
                previous,
            });
        }
        Ok(Self {
            dimension,
            scalar,
            memory,
            eta,
            mode,
            steps_seen,
            layers,
        })
    }
}

fn parse_kernel_value(value: json::Value, dimension: usize, label: &str) -> Result<ParsedKernel> {
    let mut values = parse_kernel_list(vec![value], dimension, label)?;
    Ok(values.remove(0))
}

fn parse_kernel_list(
    values: Vec<json::Value>,
    dimension: usize,
    label: &str,
) -> Result<Vec<ParsedKernel>> {
    let mut out = Vec::with_capacity(values.len());
    for (i, raw) in values.into_iter().enumerate() {
        let mut obj = expect_object(raw, &format!("{label}[{i}]"))?;
        exact_keys(&obj, &["W", "C", "V"], &format!("{label}[{i}]"))?;
        let weight = take_f64(&mut obj, "W", "W")?;
        if weight <= 0.0 {
            return Err(Error::Invalid(format!("{label}[{i}].W must be positive")));
        }
        let raw_center = take_array(&mut obj, "C", "C")?;
        if raw_center.len() != dimension {
            return Err(Error::Invalid(format!(
                "{label}[{i}].C has wrong dimension"
            )));
        }
        let mut center = Vec::with_capacity(dimension);
        for value in raw_center {
            center.push(expect_f64(value, "center component")?);
        }
        let variance = take_f64(&mut obj, "V", "V")?;
        if variance < 0.0 {
            return Err(Error::Invalid(format!(
                "{label}[{i}].V must be nonnegative"
            )));
        }
        out.push(ParsedKernel {
            weight,
            center,
            variance,
        });
    }
    Ok(out)
}

fn load_kernel_list<S: Scalar>(
    values: Vec<ParsedKernel>,
    dimension: usize,
    label: &str,
) -> Result<Vec<Kernel<S>>> {
    let mut out = Vec::with_capacity(values.len());
    for (i, raw) in values.into_iter().enumerate() {
        if raw.center.len() != dimension {
            return Err(Error::Invalid(format!("{label}[{i}] has wrong dimension")));
        }
        let projected = project_kernel::<S>(Kernel64 {
            weight: raw.weight,
            center: raw.center.clone(),
            variance: raw.variance,
        })?;
        if projected.weight.to_f64() != raw.weight
            || projected.variance.to_f64() != raw.variance
            || scalar_vec_to_f64(&projected.center) != raw.center
        {
            return Err(Error::Invalid(format!(
                "{label}[{i}] is not exactly representable in state.scalar"
            )));
        }
        out.push(projected);
    }
    sort_kernels(&mut out);
    assert_unique_geometry(&out, label)?;
    Ok(out)
}

fn load_optional_kernel<S: Scalar>(
    value: Option<ParsedKernel>,
    dimension: usize,
    label: &str,
) -> Result<Option<Kernel<S>>> {
    match value {
        None => Ok(None),
        Some(value) => {
            let mut kernels = load_kernel_list::<S>(vec![value], dimension, label)?;
            Ok(kernels.pop())
        }
    }
}

fn project_kernel<S: Scalar>(kernel: Kernel64) -> Result<Kernel<S>> {
    if !kernel.weight.is_finite() || kernel.weight <= 0.0 {
        return Err(Error::Invalid(
            "persistent kernel support is not positive".into(),
        ));
    }
    if !kernel.variance.is_finite() || kernel.variance < 0.0 {
        return Err(Error::Invalid(
            "persistent kernel variance is invalid".into(),
        ));
    }
    let mut weight = S::from_f64(kernel.weight)?;
    if weight.to_f64() <= 0.0 {
        weight = S::min_positive();
    }
    let mut center = Vec::with_capacity(kernel.center.len());
    for x in kernel.center {
        center.push(S::from_f64(x)?);
    }
    let variance = S::from_f64(kernel.variance)?;
    if variance.to_f64() < 0.0 {
        return Err(Error::Invalid(
            "persistent kernel variance is negative".into(),
        ));
    }
    let norm2 = norm2_scalar(&center);
    Ok(Kernel {
        weight,
        center,
        variance,
        norm2,
        dirty: false,
        decay_clock: None,
        decay_epoch: 0,
    })
}

fn first_coordinate_candidate_range<S: Scalar>(
    kernels: &[Kernel<S>],
    x0: f64,
    d0: f64,
) -> std::ops::Range<usize> {
    if kernels.len() <= 1 || !d0.is_finite() || d0 <= 0.0 {
        return 0..kernels.len();
    }

    // Kernels are already in canonical lexicographic center order. Any
    // concerned kernel must satisfy (x0 - C0)^2 < ||x||^2 because its full
    // squared distance is required to be < min(||x||^2, ||C||^2 + V).
    // Locate x0, then expand only while this necessary condition can hold.
    let split = kernels.partition_point(|kernel| kernel.center[0].to_f64() < x0);
    let mut start = split;
    while start > 0 {
        let delta = x0 - kernels[start - 1].center[0].to_f64();
        if delta * delta >= d0 {
            break;
        }
        start -= 1;
    }
    let mut end = split;
    while end < kernels.len() {
        let delta = kernels[end].center[0].to_f64() - x0;
        if delta * delta >= d0 {
            break;
        }
        end += 1;
    }
    start..end
}

fn point_concern_scalar<S: Scalar>(current: &[f64], source: &[S]) -> bool {
    debug_assert_eq!(current.len(), source.len());
    let current2 = norm2(current);
    let source2 = norm2_scalar(source);
    let current_nonzero = current.iter().any(|&x| x != 0.0);
    let source_nonzero = source.iter().any(|&x| x.to_f64() != 0.0);
    let extreme = !current2.is_finite()
        || !source2.is_finite()
        || (current2 == 0.0 && current_nonzero)
        || (source2 == 0.0 && source_nonzero);

    if extreme {
        let mut scale = 0.0f64;
        for &x in current {
            scale = scale.max(x.abs());
        }
        for &x in source {
            scale = scale.max(x.to_f64().abs());
        }
        if scale == 0.0 {
            return false;
        }
        let current_scaled = stable_sum(current.iter().map(|&x| {
            let y = x / scale;
            y * y
        }));
        let source_scaled = stable_sum(source.iter().map(|&x| {
            let y = x.to_f64() / scale;
            y * y
        }));
        let distance_scaled = stable_sum(current.iter().zip(source).map(|(&a, &b)| {
            let d = a / scale - b.to_f64() / scale;
            d * d
        }));
        return distance_scaled < current_scaled && distance_scaled < source_scaled;
    }

    let distance2 = stable_sum(current.iter().zip(source).map(|(&a, &b)| {
        let d = a - b.to_f64();
        d * d
    }));
    distance2 < current2 && distance2 < source2
}

fn concern_scalar<S: Scalar>(
    kernel: &Kernel<S>,
    x: &[f64],
    input_variance: f64,
    d0_center: f64,
) -> (bool, f64, f64) {
    // A finite nonzero vector may have a squared norm that overflows or
    // underflows binary64. In that representational corner, evaluate the
    // same homogeneous inequalities in units of the incoming atom. All gains
    // for this atom then share the same positive scale factor, so ALLOCATE is
    // unchanged.
    if !d0_center.is_finite() || (d0_center == 0.0 && x.iter().any(|&v| v != 0.0)) {
        let mut scale = if input_variance > 0.0 {
            input_variance.sqrt()
        } else {
            0.0
        };
        for &value in x {
            scale = scale.max(value.abs());
        }
        if scale == 0.0 {
            return (false, 0.0, 0.0);
        }
        let x2 = stable_sum(x.iter().map(|&value| {
            let y = value / scale;
            y * y
        }));
        let geometric = stable_sum(x.iter().zip(&kernel.center).map(|(&a, &b)| {
            let d = a / scale - b.to_f64() / scale;
            d * d
        }));
        if geometric.partial_cmp(&x2) != Some(std::cmp::Ordering::Less) {
            return (false, x2 - geometric, geometric);
        }
        let vin = if input_variance > 0.0 {
            let y = input_variance.sqrt() / scale;
            y * y
        } else {
            0.0
        };
        let kernel_norm = stable_sum(kernel.center.iter().map(|&value| {
            let y = value.to_f64() / scale;
            y * y
        }));
        let kernel_variance = if kernel.variance.to_f64() > 0.0 {
            let y = kernel.variance.to_f64().sqrt() / scale;
            y * y
        } else {
            0.0
        };
        let ok = geometric + vin < kernel_norm + kernel_variance;
        let distance2 = if geometric == 0.0 {
            0.0
        } else {
            let distance = geometric.sqrt() * scale;
            distance * distance
        };
        return (ok, x2 - geometric, distance2);
    }

    // The first canonical bound is D_i < D_0 with the same incoming V on
    // both sides. Geometrically this still requires ||x-C||^2 < ||x||^2,
    // which remains a safe early-stop criterion. The second bound compares
    // the full incoming contextual dispersion.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for (&a, &b) in x.iter().zip(&kernel.center) {
        let d = a - b.to_f64();
        let term = d * d;
        if term >= d0_center {
            return (false, 0.0, 0.0);
        }
        compensated_add(&mut sum, &mut correction, term);
    }
    let geometric = compensated_finish(sum, correction);
    let da = structural_zero(geometric + input_variance);
    let d0 = structural_zero(d0_center + input_variance);
    let ok = da < d0 && da < kernel.norm2 + kernel.variance.to_f64();
    (ok, d0_center - geometric, geometric)
}

fn concern_kernel<S: Scalar>(kernel: &Kernel<S>, x: &Kernel<S>) -> (bool, f64) {
    concern_raw_kernel(
        kernel,
        &scalar_vec_to_f64(&x.center),
        x.variance.to_f64(),
        x.norm2,
    )
}

fn concern_raw_kernel<S: Scalar>(
    kernel: &Kernel<S>,
    center: &[f64],
    variance: f64,
    center_norm2: f64,
) -> (bool, f64) {
    let (ok, gain, _) = concern_scalar(kernel, center, variance, center_norm2);
    (ok, gain)
}

fn coalesce_atoms(mut atoms: Vec<Atom>) -> Vec<Atom> {
    atoms.sort_by(|a, b| {
        cmp_vec(a.x.as_ref(), b.x.as_ref()).then_with(|| a.variance.total_cmp(&b.variance))
    });
    if !atoms
        .windows(2)
        .any(|pair| pair[0].x == pair[1].x && pair[0].variance == pair[1].variance)
    {
        return atoms;
    }
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        let x = atoms[i].x.clone();
        let variance = atoms[i].variance;
        let mut j = i + 1;
        while j < atoms.len() && atoms[j].x == x && atoms[j].variance == variance {
            j += 1;
        }
        out.push(Atom {
            x,
            r: stable_sum(atoms[i..j].iter().map(|atom| atom.r)),
            variance,
            norm2: atoms[i].norm2,
            zero: atoms[i].zero,
        });
        i = j;
    }
    out
}

fn build_context_kernel<S: Scalar>(
    cells: &[Kernel<S>],
    contributions: &[(usize, f64)],
    dimension: usize,
) -> Option<Kernel64> {
    if contributions.is_empty() {
        return None;
    }
    let mut terms = Vec::with_capacity(contributions.len().max(dimension));
    let mut variance_terms = Vec::with_capacity(contributions.len());
    let mut partials = Vec::new();

    terms.extend(contributions.iter().map(|&(_, weight)| weight));
    let weight = orderless_sum(&terms, &mut partials);
    let first_center = &cells[contributions[0].0].center;
    if contributions[1..]
        .iter()
        .all(|&(index, _)| cells[index].center == *first_center)
    {
        return Some(Kernel64 {
            weight,
            center: first_center.iter().map(|&x| x.to_f64()).collect(),
            variance: 0.0,
        });
    }
    let mut center = vec![0.0; dimension];
    for (j, component) in center.iter_mut().enumerate() {
        terms.clear();
        terms.extend(
            contributions
                .iter()
                .map(|&(index, w)| w * cells[index].center[j].to_f64()),
        );
        *component = structural_zero(orderless_sum(&terms, &mut partials) / weight);
    }

    for &(index, w) in contributions {
        terms.clear();
        terms.extend(cells[index].center.iter().zip(&center).map(|(&x, &c)| {
            let d = x.to_f64() - c;
            d * d
        }));
        let distance2 = orderless_sum(&terms, &mut partials);
        variance_terms.push(w * distance2);
    }
    let variance = structural_zero(orderless_sum(&variance_terms, &mut partials) / weight);
    Some(Kernel64 {
        weight,
        center,
        variance,
    })
}

fn coalesce_kernel64(mut kernels: Vec<Kernel64>) -> Vec<Kernel64> {
    kernels.sort_by(|a, b| {
        cmp_vec(&a.center, &b.center).then_with(|| a.variance.total_cmp(&b.variance))
    });
    if !kernels
        .windows(2)
        .any(|pair| pair[0].center == pair[1].center && pair[0].variance == pair[1].variance)
    {
        return kernels;
    }
    let mut out = Vec::with_capacity(kernels.len());
    let mut i = 0;
    while i < kernels.len() {
        let center = kernels[i].center.clone();
        let variance = kernels[i].variance;
        let mut j = i + 1;
        while j < kernels.len() && kernels[j].center == center && kernels[j].variance == variance {
            j += 1;
        }
        out.push(Kernel64 {
            weight: stable_sum(kernels[i..j].iter().map(|k| k.weight)),
            center,
            variance,
        });
        i = j;
    }
    out
}

fn retain_touched_nonzero<S: Scalar>(kernels: &mut Vec<Kernel<S>>, touched: &[usize]) -> bool {
    let removes_any = touched
        .iter()
        .copied()
        .any(|index| kernels[index].dirty && zero_scalar_vec(&kernels[index].center));
    if !removes_any {
        return true;
    }
    kernels.retain(|kernel| !kernel.dirty || !zero_scalar_vec(&kernel.center));
    false
}

fn coalesce_touched<S: Scalar>(
    kernels: Vec<Kernel<S>>,
    touched: &[usize],
    indices_stable: bool,
) -> Result<(Vec<Kernel<S>>, bool)> {
    if kernels.len() <= 1 {
        return Ok((kernels, indices_stable));
    }
    if indices_stable && touched.len() < kernels.len() {
        let mut local_order_ok = true;
        for &index in touched {
            if index > 0
                && kernel_geometry_cmp(&kernels[index - 1], &kernels[index]) != Ordering::Less
            {
                local_order_ok = false;
                break;
            }
            if index + 1 < kernels.len()
                && kernel_geometry_cmp(&kernels[index], &kernels[index + 1]) != Ordering::Less
            {
                local_order_ok = false;
                break;
            }
        }
        if local_order_ok {
            return Ok((kernels, true));
        }
    }
    Ok((coalesce_dirty(kernels)?, false))
}

fn coalesce_dirty<S: Scalar>(mut kernels: Vec<Kernel<S>>) -> Result<Vec<Kernel<S>>> {
    if !kernels.iter().any(|kernel| kernel.dirty) {
        return Ok(kernels);
    }

    let mut ordered = true;
    let mut has_clone = false;
    for pair in kernels.windows(2) {
        let order = kernel_geometry_cmp(&pair[0], &pair[1]);
        ordered &= order != Ordering::Greater;
        has_clone |= order == Ordering::Equal;
        if !ordered {
            break;
        }
    }
    if ordered && !has_clone {
        return Ok(kernels);
    }
    if !ordered {
        kernels.sort_by(kernel_geometry_cmp);
    }
    if !has_clone
        && !kernels
            .windows(2)
            .any(|pair| pair[0].center == pair[1].center && pair[0].variance == pair[1].variance)
    {
        return Ok(kernels);
    }
    let mut out = Vec::with_capacity(kernels.len());
    let mut i = 0;
    while i < kernels.len() {
        let center = kernels[i].center.clone();
        let variance = kernels[i].variance;
        let mut dirty = kernels[i].dirty;
        let mut j = i + 1;
        while j < kernels.len() && kernels[j].center == center && kernels[j].variance == variance {
            dirty |= kernels[j].dirty;
            j += 1;
        }
        let mut weight = S::from_f64(stable_sum(kernels[i..j].iter().map(Kernel::weight)))?;
        if weight.to_f64() <= 0.0 {
            weight = S::min_positive();
        }
        let norm2 = kernels[i].norm2;
        out.push(Kernel {
            weight,
            center,
            variance,
            norm2,
            dirty,
            decay_clock: kernels[i].decay_clock.clone(),
            decay_epoch: kernels[i]
                .decay_clock
                .as_ref()
                .map_or(0, |clock| clock.epoch()),
        });
        i = j;
    }
    Ok(out)
}

fn kernel_geometry_cmp<S: Scalar>(a: &Kernel<S>, b: &Kernel<S>) -> Ordering {
    cmp_scalar_vec(&a.center, &b.center)
        .then_with(|| a.variance.to_f64().total_cmp(&b.variance.to_f64()))
}

fn coalesce_projected<S: Scalar>(mut kernels: Vec<Kernel<S>>) -> Result<Vec<Kernel<S>>> {
    sort_kernels(&mut kernels);
    if !kernels
        .windows(2)
        .any(|pair| pair[0].center == pair[1].center && pair[0].variance == pair[1].variance)
    {
        return Ok(kernels);
    }
    let mut out = Vec::with_capacity(kernels.len());
    let mut i = 0;
    while i < kernels.len() {
        let center = kernels[i].center.clone();
        let variance = kernels[i].variance;
        let mut j = i + 1;
        while j < kernels.len() && kernels[j].center == center && kernels[j].variance == variance {
            j += 1;
        }
        let mut weight = S::from_f64(stable_sum(
            kernels[i..j].iter().map(|kernel| kernel.weight.to_f64()),
        ))?;
        if weight.to_f64() <= 0.0 {
            weight = S::min_positive();
        }
        let norm2 = kernels[i].norm2;
        out.push(Kernel {
            weight,
            center,
            variance,
            norm2,
            dirty: false,
            decay_clock: None,
            decay_epoch: 0,
        });
        i = j;
    }
    Ok(out)
}

fn sort_kernels<S: Scalar>(kernels: &mut [Kernel<S>]) {
    kernels.sort_by(kernel_geometry_cmp);
}

fn assert_unique_geometry<S: Scalar>(kernels: &[Kernel<S>], label: &str) -> Result<()> {
    for pair in kernels.windows(2) {
        if pair[0].center == pair[1].center && pair[0].variance == pair[1].variance {
            return Err(Error::Invalid(format!(
                "{label} contains uncoalesced exact clones"
            )));
        }
    }
    Ok(())
}

#[inline]
fn compensated_add(sum: &mut f64, correction: &mut f64, x: f64) {
    let next = *sum + x;
    if sum.abs() >= x.abs() {
        *correction += (*sum - next) + x;
    } else {
        *correction += (x - next) + *sum;
    }
    *sum = next;
}

#[inline]
fn compensated_finish(sum: f64, correction: f64) -> f64 {
    structural_zero(sum + correction)
}

fn stable_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    // Neumaier compensated sum. Inputs are consumed in canonical order by all
    // behavioral callers; compensation keeps the result close to exact while
    // remaining compact and dependency-free.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for x in values {
        compensated_add(&mut sum, &mut correction, x);
    }
    compensated_finish(sum, correction)
}

fn orderless_sum(values: &[f64], partials: &mut Vec<f64>) -> f64 {
    partials.clear();
    for &value in values {
        let mut x = value;
        let mut write = 0usize;
        let len = partials.len();
        for read in 0..len {
            let mut y = partials[read];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[write] = lo;
                write += 1;
            }
            x = hi;
        }
        partials.truncate(write);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let mut out = 0.0;
    for &x in partials.iter() {
        out += x;
    }
    structural_zero(out)
}

fn norm2(v: &[f64]) -> f64 {
    stable_sum(v.iter().map(|x| x * x))
}

fn norm2_scalar<S: Scalar>(v: &[S]) -> f64 {
    stable_sum(v.iter().map(|x| {
        let x = x.to_f64();
        x * x
    }))
}

fn zero_f64_vec(v: &[f64]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

fn zero_scalar_vec<S: Scalar>(v: &[S]) -> bool {
    v.iter().all(|&x| x.to_f64() == 0.0)
}

fn scalar_vec_to_f64<S: Scalar>(v: &[S]) -> Vec<f64> {
    v.iter().map(|&x| x.to_f64()).collect()
}

fn structural_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn cmp_vec(a: &[f64], b: &[f64]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let ord = x.total_cmp(y);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

fn cmp_scalar_vec<S: Scalar>(a: &[S], b: &[S]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let ord = x.to_f64().total_cmp(&y.to_f64());
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

fn recognition_cmp(a: &Recognition, b: &Recognition) -> Ordering {
    let input_order = if Arc::ptr_eq(&a.local_input, &b.local_input) {
        Ordering::Equal
    } else {
        cmp_vec(a.local_input.as_ref(), b.local_input.as_ref())
    };
    input_order.then_with(|| cmp_vec(&a.recognised, &b.recognised))
}

fn dedup_single_layer_recognitions(readout: &mut Vec<Recognition>) {
    readout.dedup_by(|a, b| {
        Arc::ptr_eq(&a.local_input, &b.local_input) && a.recognised == b.recognised
    });
}

fn dedup_recognitions(readout: &mut Vec<Recognition>) {
    readout.dedup_by(|a, b| {
        (Arc::ptr_eq(&a.local_input, &b.local_input) || a.local_input == b.local_input)
            && a.recognised == b.recognised
    });
}

fn sort_dedup_recognitions(readout: &mut Vec<Recognition>) {
    if readout
        .windows(2)
        .any(|pair| recognition_cmp(&pair[0], &pair[1]) == Ordering::Greater)
    {
        readout.sort_by(recognition_cmp);
    }
    dedup_recognitions(readout);
}

fn temporal_recognition_cmp(a: &TemporalRecognition, b: &TemporalRecognition) -> Ordering {
    a.universe
        .as_ref()
        .cmp(b.universe.as_ref())
        .then_with(|| cmp_vec(a.previous_input.as_ref(), b.previous_input.as_ref()))
        .then_with(|| cmp_vec(a.current_input.as_ref(), b.current_input.as_ref()))
        .then_with(|| cmp_vec(&a.previous_recognised, &b.previous_recognised))
        .then_with(|| cmp_vec(&a.current_recognised, &b.current_recognised))
}

fn sort_dedup_temporal_recognitions(readout: &mut Vec<TemporalRecognition>) {
    readout.sort_by(temporal_recognition_cmp);
    readout.dedup_by(|a, b| {
        a.universe == b.universe
            && a.previous_input == b.previous_input
            && a.current_input == b.current_input
            && a.previous_recognised == b.previous_recognised
            && a.current_recognised == b.current_recognised
    });
}

fn prediction_cmp(a: &Prediction, b: &Prediction) -> Ordering {
    a.universe
        .cmp(&b.universe)
        .then_with(|| cmp_vec(&a.current_context, &b.current_context))
        .then_with(|| cmp_vec(&a.recognised_source, &b.recognised_source))
        .then_with(|| cmp_vec(&a.predicted_successor, &b.predicted_successor))
}

fn sort_dedup_predictions(readout: &mut Vec<Prediction>) {
    readout.sort_by(prediction_cmp);
    readout.dedup_by(|a, b| {
        a.universe == b.universe
            && a.current_context == b.current_context
            && a.recognised_source == b.recognised_source
            && a.predicted_successor == b.predicted_successor
    });
}

fn ratio_string(numerator: u64, denominator: u64) -> Result<String> {
    if denominator == 0 {
        return Err(Error::Invalid("zero budget unit".into()));
    }
    let whole = numerator / denominator;
    let mut rem = (numerator % denominator) as u128;
    let denominator_u128 = denominator as u128;
    if rem == 0 {
        return Ok(whole.to_string());
    }
    // Decimal if terminating, otherwise exact rational. Both are lossless.
    let mut d = denominator;
    while d % 2 == 0 {
        d /= 2;
    }
    while d % 5 == 0 {
        d /= 5;
    }
    if d != 1 {
        return Ok(format!("{numerator}/{denominator}"));
    }
    let mut out = format!("{whole}.");
    while rem != 0 {
        rem *= 10;
        out.push(char::from(b'0' + (rem / denominator_u128) as u8));
        rem %= denominator_u128;
    }
    Ok(out)
}

fn push_float(out: &mut String, value: f64) {
    if value == 0.0 {
        out.push_str("0.0");
    } else {
        out.push_str(&value.to_string());
    }
}

fn write_kernel_list<S: Scalar>(out: &mut String, kernels: &[Kernel<S>]) {
    for (i, kernel) in kernels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"W\":");
        push_float(out, kernel.weight());
        out.push_str(",\"C\":[");
        for (j, x) in kernel.center.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            push_float(out, x.to_f64());
        }
        out.push_str("],\"V\":");
        push_float(out, kernel.variance.to_f64());
        out.push('}');
    }
}

fn write_kernel<S: Scalar>(out: &mut String, kernel: &Kernel<S>) {
    out.push_str("{\"W\":");
    push_float(out, kernel.weight());
    out.push_str(",\"C\":[");
    for (j, x) in kernel.center.iter().enumerate() {
        if j > 0 {
            out.push(',');
        }
        push_float(out, x.to_f64());
    }
    out.push_str("],\"V\":");
    push_float(out, kernel.variance.to_f64());
    out.push('}');
}

fn expect_object(value: json::Value, label: &str) -> Result<BTreeMap<String, json::Value>> {
    match value {
        json::Value::Object(v) => Ok(v),
        _ => Err(Error::Invalid(format!("{label} must be an object"))),
    }
}

fn exact_keys(obj: &BTreeMap<String, json::Value>, expected: &[&str], label: &str) -> Result<()> {
    let mut actual: Vec<&str> = obj.keys().map(String::as_str).collect();
    actual.sort_unstable();
    let mut want = expected.to_vec();
    want.sort_unstable();
    if actual != want {
        return Err(Error::Invalid(format!(
            "{label} has missing or unknown keys"
        )));
    }
    Ok(())
}

fn take_u64(obj: &mut BTreeMap<String, json::Value>, key: &str, label: &str) -> Result<u64> {
    let value = obj
        .remove(key)
        .ok_or_else(|| Error::Invalid(format!("missing {label}")))?;
    match value {
        json::Value::Number(text) if !text.contains(['.', 'e', 'E']) && !text.starts_with('-') => {
            text.parse::<u64>()
                .map_err(|_| Error::Invalid(format!("{label} must be a nonnegative integer")))
        }
        _ => Err(Error::Invalid(format!(
            "{label} must be a nonnegative integer"
        ))),
    }
}

fn take_f64(obj: &mut BTreeMap<String, json::Value>, key: &str, label: &str) -> Result<f64> {
    let value = obj
        .remove(key)
        .ok_or_else(|| Error::Invalid(format!("missing {label}")))?;
    expect_f64(value, label)
}

fn expect_f64(value: json::Value, label: &str) -> Result<f64> {
    match value {
        json::Value::Number(text) => {
            let out: f64 = text
                .parse()
                .map_err(|_| Error::Invalid(format!("{label} must be a real number")))?;
            if !out.is_finite() {
                return Err(Error::Invalid(format!("{label} must be finite")));
            }
            Ok(structural_zero(out))
        }
        _ => Err(Error::Invalid(format!("{label} must be a real number"))),
    }
}

fn take_string(obj: &mut BTreeMap<String, json::Value>, key: &str, label: &str) -> Result<String> {
    match obj.remove(key) {
        Some(json::Value::String(v)) => Ok(v),
        _ => Err(Error::Invalid(format!("{label} must be a string"))),
    }
}

fn take_array(
    obj: &mut BTreeMap<String, json::Value>,
    key: &str,
    label: &str,
) -> Result<Vec<json::Value>> {
    match obj.remove(key) {
        Some(json::Value::Array(v)) => Ok(v),
        _ => Err(Error::Invalid(format!("{label} must be a list"))),
    }
}

pub fn parse_presentation_json(text: &str) -> Result<Vec<Vec<f64>>> {
    let value = json::parse(text)?;
    let json::Value::Array(items) = value else {
        return Err(Error::Invalid(
            "presentation must be a JSON array of vectors".into(),
        ));
    };
    if items.is_empty() {
        return Err(Error::Invalid("presentation must be nonempty".into()));
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let json::Value::Array(coords) = item else {
            return Err(Error::Invalid("presentation items must be vectors".into()));
        };
        let mut x = Vec::with_capacity(coords.len());
        for c in coords {
            x.push(expect_f64(c, "vector component")?);
        }
        out.push(x);
    }
    Ok(out)
}

fn write_temporal_recognition_items(out: &mut String, sequences: &[TemporalRecognition]) {
    for (i, rec) in sequences.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        json::quote(out, &rec.universe);
        out.push_str(",[");
        write_f64_vec(out, rec.previous_input.as_ref());
        out.push(',');
        write_f64_vec(out, rec.current_input.as_ref());
        out.push_str("],[");
        write_f64_vec(out, &rec.previous_recognised);
        out.push(',');
        write_f64_vec(out, &rec.current_recognised);
        out.push_str("]]");
    }
}

pub fn step_report_json(report: &StepReport) -> String {
    let mut out = String::with_capacity(768);
    out.push('{');
    out.push_str("\"step_index\":");
    out.push_str(&report.step_index.to_string());
    out.push_str(",\"readout\":");
    match &report.readout {
        Readout::Geometry(concepts) => write_recognition_list(&mut out, concepts),
        Readout::Temporal {
            concepts,
            sequences,
        } => {
            out.push_str("{\"concepts\":");
            write_recognition_list(&mut out, concepts);
            out.push_str(",\"sequences\":[");
            write_temporal_recognition_items(&mut out, sequences);
            out.push_str("]}");
        }
        Readout::Predictive {
            concepts,
            sequences,
            predictions,
        } => {
            out.push_str("{\"concepts\":");
            write_recognition_list(&mut out, concepts);
            out.push_str(",\"sequences\":[");
            write_temporal_recognition_items(&mut out, sequences);
            out.push_str("],\"predictions\":[");
            for (i, prediction) in predictions.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                json::quote(&mut out, &prediction.universe);
                out.push(',');
                write_f64_vec(&mut out, &prediction.current_context);
                out.push(',');
                write_f64_vec(&mut out, &prediction.recognised_source);
                out.push(',');
                write_f64_vec(&mut out, &prediction.predicted_successor);
                out.push(']');
            }
            out.push_str("]}");
        }
    }
    out.push_str(",\"transformations\":[");
    for (i, t) in report.transformations.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_transformation(&mut out, t);
    }
    out.push_str("],\"maintenance_open_units\":");
    out.push_str(&report.maintenance_open_units.to_string());
    out.push_str(",\"maintenance_units\":");
    out.push_str(&report.maintenance_units.to_string());
    out.push_str(",\"budget_units\":");
    out.push_str(&report.budget_units.to_string());
    out.push_str(",\"layer_reports\":[");
    for (i, r) in report.layer_reports.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_layer_report(&mut out, r);
    }
    out.push_str("],\"temporal_reports\":[");
    for (i, r) in report.temporal_reports.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_layer_report(&mut out, r);
    }
    out.push_str("]}");
    out
}

fn write_recognition_list(out: &mut String, values: &[Recognition]) {
    out.push('[');
    for (i, rec) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        json::quote(out, &rec.universe);
        out.push(',');
        write_f64_vec(out, rec.local_input.as_ref());
        out.push(',');
        write_f64_vec(out, &rec.recognised);
        out.push(']');
    }
    out.push(']');
}

pub fn summary_json(summary: &Summary) -> String {
    let mut out = String::with_capacity(640);
    out.push('{');
    out.push_str("\"steps_seen\":");
    out.push_str(&summary.steps_seen.to_string());
    out.push_str(",\"dimension\":");
    out.push_str(&summary.dimension.to_string());
    out.push_str(",\"universe\":");
    json::quote(&mut out, &summary.universe);
    out.push_str(",\"scalar\":");
    json::quote(&mut out, summary.scalar);
    out.push_str(",\"memory\":");
    push_float(&mut out, summary.memory);
    out.push_str(",\"eta\":");
    push_float(&mut out, summary.eta);
    out.push_str(",\"mode\":");
    json::quote(&mut out, summary.mode.as_str());
    out.push_str(",\"chi\":");
    push_float(&mut out, summary.chi);
    out.push_str(",\"alpha\":");
    push_float(&mut out, summary.alpha);
    out.push_str(",\"effective_alpha\":");
    push_float(&mut out, summary.effective_alpha);
    out.push_str(",\"layer_count\":");
    out.push_str(&summary.layer_count.to_string());
    out.push_str(",\"cells_per_layer\":");
    write_usize_vec(&mut out, &summary.cells_per_layer);
    out.push_str(",\"sigma_per_layer\":");
    write_usize_vec(&mut out, &summary.sigma_per_layer);
    out.push_str(",\"temporal_cells_per_layer\":");
    write_usize_vec(&mut out, &summary.temporal_cells_per_layer);
    out.push_str(",\"temporal_sigma_per_layer\":");
    write_usize_vec(&mut out, &summary.temporal_sigma_per_layer);
    out.push_str(",\"previous_context_per_layer\":[");
    for (i, value) in summary.previous_context_per_layer.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(if *value { "true" } else { "false" });
    }
    out.push_str("],\"maintenance_units\":");
    out.push_str(&summary.maintenance_units.to_string());
    out.push_str(",\"budget\":");
    json::quote(&mut out, &summary.budget);
    out.push_str(",\"budget_units\":");
    out.push_str(&summary.budget_units.to_string());
    out.push_str(",\"budget_margin_units\":");
    out.push_str(&summary.budget_margin_units.to_string());
    out.push_str(",\"is_solvent\":");
    out.push_str(if summary.is_solvent { "true" } else { "false" });
    out.push('}');
    out
}

fn write_f64_vec(out: &mut String, values: &[f64]) {
    out.push('[');
    for (i, x) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_float(out, *x);
    }
    out.push(']');
}

fn write_usize_vec(out: &mut String, values: &[usize]) {
    out.push('[');
    for (i, x) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&x.to_string());
    }
    out.push(']');
}

fn write_transformation(out: &mut String, t: &Transformation) {
    match t {
        Transformation::ClearSigma { count } => {
            out.push_str("{\"phase\":\"solvency\",\"type\":\"clear_sigma\",\"count\":");
            out.push_str(&count.to_string());
            out.push('}');
        }
        Transformation::TrimLayers { count } => {
            out.push_str("{\"phase\":\"solvency\",\"type\":\"trim_layers\",\"count\":");
            out.push_str(&count.to_string());
            out.push('}');
        }
        Transformation::DestroyCells {
            count,
            waves,
            k_through,
        } => {
            out.push_str("{\"phase\":\"solvency\",\"type\":\"destroy_cells\",\"count\":");
            out.push_str(&count.to_string());
            out.push_str(",\"waves\":");
            out.push_str(&waves.to_string());
            out.push_str(",\"K_through\":");
            push_float(out, *k_through);
            out.push('}');
        }
        Transformation::Promote {
            space,
            layer,
            count,
        } => {
            out.push_str("{\"phase\":");
            json::quote(out, space.as_str());
            out.push_str(",\"type\":\"promote\",\"layer\":");
            out.push_str(&layer.to_string());
            out.push_str(",\"count\":");
            out.push_str(&count.to_string());
            out.push('}');
        }
        Transformation::GrowthCommit {
            geometric_seeds,
            temporal_seeds,
            seeds,
            layer_created,
            units,
        } => {
            out.push_str("{\"phase\":\"growth\",\"type\":\"commit\",\"geometric_seeds\":");
            out.push_str(&geometric_seeds.to_string());
            out.push_str(",\"temporal_seeds\":");
            out.push_str(&temporal_seeds.to_string());
            out.push_str(",\"seeds\":");
            out.push_str(&seeds.to_string());
            out.push_str(",\"layer_created\":");
            out.push_str(if *layer_created { "true" } else { "false" });
            out.push_str(",\"units\":");
            out.push_str(&units.to_string());
            out.push('}');
        }
        Transformation::GrowthReject {
            geometric_seeds,
            temporal_seeds,
            seeds,
            layer_requested,
            units,
        } => {
            out.push_str("{\"phase\":\"growth\",\"type\":\"reject\",\"geometric_seeds\":");
            out.push_str(&geometric_seeds.to_string());
            out.push_str(",\"temporal_seeds\":");
            out.push_str(&temporal_seeds.to_string());
            out.push_str(",\"seeds\":");
            out.push_str(&seeds.to_string());
            out.push_str(",\"layer_requested\":");
            out.push_str(if *layer_requested { "true" } else { "false" });
            out.push_str(",\"units\":");
            out.push_str(&units.to_string());
            out.push('}');
        }
    }
}

fn write_layer_report(out: &mut String, r: &LayerReport) {
    out.push('{');
    macro_rules! int_field {
        ($name:literal, $value:expr) => {{
            out.push_str(concat!("\"", $name, "\":"));
            out.push_str(&$value.to_string());
        }};
    }
    out.push_str("\"phase\":");
    json::quote(out, r.phase.as_str());
    out.push(',');
    int_field!("layer_index", r.layer_index);
    out.push(',');
    int_field!("input_atom_count", r.input_atom_count);
    out.push_str(",\"input_mass\":");
    push_float(out, r.input_mass);
    out.push(',');
    int_field!("unknown_atom_count", r.unknown_atom_count);
    out.push(',');
    int_field!("recognised_atom_count", r.recognised_atom_count);
    out.push(',');
    int_field!("cell_count_before", r.cell_count_before);
    out.push(',');
    int_field!("cell_count_after", r.cell_count_after);
    out.push(',');
    int_field!("sigma_count_before", r.sigma_count_before);
    out.push(',');
    int_field!("sigma_count_after", r.sigma_count_after);
    out.push(',');
    int_field!("promoted", r.promoted);
    out.push(',');
    int_field!("seed_requests", r.seed_requests);
    out.push_str(",\"context_emitted\":");
    out.push_str(if r.context_emitted { "true" } else { "false" });
    out.push(',');
    int_field!("output_atom_count", r.output_atom_count);
    out.push_str(",\"output_mass\":");
    push_float(out, r.output_mass);
    out.push_str(",\"context_center\":");
    match &r.context_center {
        Some(center) => write_f64_vec(out, center),
        None => out.push_str("null"),
    }
    out.push_str(",\"context_variance\":");
    match r.context_variance {
        Some(variance) => push_float(out, variance),
        None => out.push_str("null"),
    }
    out.push(',');
    int_field!("recognition_count", r.recognition_count);
    out.push_str(",\"cell_responsibility_mass\":");
    write_f64_vec(out, &r.cell_responsibility_mass);
    out.push('}');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make64() -> Auxein<f64> {
        Auxein::new(1, 10.0, 1.0, Budget::kernels("100"), "auxein").unwrap()
    }

    fn make_temporal64() -> Auxein<f64> {
        Auxein::new_with_mode(
            1,
            10.0,
            1.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "auxein",
        )
        .unwrap()
    }

    fn make_predictive64() -> Auxein<f64> {
        Auxein::new_with_mode(
            1,
            10.0,
            1.0,
            Mode::Predictive,
            Budget::kernels("100"),
            "auxein",
        )
        .unwrap()
    }

    #[test]
    fn packing() {
        let n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("0"), "auxein").unwrap();
        assert_eq!(n.kernel_units().unwrap(), 24);
        assert_eq!(n.network_units().unwrap(), 50);
        assert_eq!(n.min_units().unwrap(), 66);
        assert_eq!(n.budget_units(), 66);
        assert_eq!(n.maintenance_units().unwrap(), 66);
    }

    #[test]
    fn extreme_f64_exact_recurrence_remains_recognisable() {
        for x in [f64::from_bits(1), 1e-200, 1e200, 1e308] {
            let mut n = make64();
            n.step(&[vec![x]], false).unwrap();
            n.step(&[vec![x]], false).unwrap();
            let report = n.step(&[vec![x]], false).unwrap();
            assert_eq!(report.readout.len(), 1);
            assert_eq!(report.readout[0].local_input.as_ref(), &[x]);
            assert_eq!(report.readout[0].recognised, vec![x]);
        }
    }

    #[test]
    fn f64_support_underflow_is_not_cognitive_death() {
        let mut n = Auxein::<f64>::new(1, 0.25, 1.0, Budget::kernels("1000"), "auxein").unwrap();
        n.step(&[vec![10.0]], false).unwrap();
        for _ in 0..320 {
            n.step(&[vec![1.0]], false).unwrap();
        }
        let old = n.layers()[0]
            .sigma()
            .iter()
            .find(|kernel| kernel.center() == [10.0])
            .unwrap();
        assert_eq!(old.weight(), f64::from_bits(1));
    }

    #[test]
    fn first_then_recurrence_stays_local_until_context_exists() {
        let mut n = make64();
        let r1 = n.step(&[vec![2.0]], false).unwrap();
        assert!(r1.readout.is_empty());
        assert_eq!(n.layers()[0].sigma().len(), 1);
        let r2 = n.step(&[vec![2.0]], true).unwrap();
        assert!(r2.readout.is_empty());
        assert_eq!(n.layers()[0].cells().len(), 1);
        assert_eq!(n.layers().len(), 1);
        let r3 = n.step(&[vec![2.0]], true).unwrap();
        assert_eq!(r3.readout.len(), 1);
        assert_eq!(r3.readout[0].recognised, vec![2.0]);
        assert!(!r3.layer_reports[0].context_emitted);
        assert_eq!(n.layers().len(), 1);
    }

    #[test]
    fn eta_zero_freezes_but_recognises() {
        let mut n = make64();
        n.step(&[vec![2.0]], false).unwrap();
        n.step(&[vec![2.0]], false).unwrap();
        n.step(&[vec![2.0]], false).unwrap();
        n.set_eta(0.0).unwrap();
        let before = n.export_json();
        let report = n.step(&[vec![2.0]], false).unwrap();
        assert!(!report.readout.is_empty());
        let after = n.export_json();
        assert_ne!(before, after); // steps_seen changes
        let restored =
            Auxein::<f64>::from_json(&after, Budget::units(n.budget_units()), "auxein").unwrap();
        assert_eq!(restored.export_json(), after);
    }

    #[test]
    fn duplicate_coalescence() {
        let mut a = make64();
        let mut b = make64();
        for _ in 0..3 {
            a.step(&[vec![2.0]], false).unwrap();
            b.step(&[vec![2.0], vec![2.0], vec![2.0], vec![2.0]], false)
                .unwrap();
        }
        assert_eq!(a.export_json(), b.export_json());
    }

    #[test]
    fn whole_presentation_splitting_is_exactly_invariant() {
        let mut a = Auxein::<f32>::new(3, 10.0, 1.0, Budget::kernels("100"), "auxein").unwrap();
        let mut b = a.clone();
        let p = vec![
            vec![6.125, -5.75, 6.375],
            vec![6.125, -5.75, 6.375],
            vec![-3.875, -1.75, -5.625],
            vec![5.1259765625, 7.2509765625, -3.6250009536743164],
        ];
        let mut split = Vec::new();
        for _ in 0..3 {
            split.extend(p.iter().cloned());
        }
        for _ in 0..50 {
            let ra = a.step(&p, false).unwrap();
            let rb = b.step(&split, false).unwrap();
            assert_eq!(step_report_json(&ra), step_report_json(&rb));
            assert_eq!(a.export_json(), b.export_json());
        }
    }

    #[test]
    fn f32_state_roundtrip() {
        let n = Auxein::<f32>::new(1, 10.1, 0.7, Budget::kernels("100"), "x").unwrap();
        assert_ne!(n.memory(), 10.1);
        let state = n.export_json();
        let restored =
            Auxein::<f32>::from_json(&state, Budget::units(n.budget_units()), "x").unwrap();
        assert_eq!(restored.export_json(), state);
    }

    #[test]
    fn f32_projected_seed_is_revalidated_before_persistence() {
        let state = r#"{"format_version":4,"dimension":2,"scalar":"f32","memory":1.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0,1.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f32>::from_json(state, Budget::kernels("100"), "projection").unwrap();

        // In binary64 this point is just outside the strict first CONCERN
        // bound x+y>1. Its f32 projection lies just inside it. A raw seed may
        // therefore be requested, but the projected kernel must be
        // revalidated before it can become persistent Sigma state.
        let report = n.step(&[vec![0.199999999, 0.8]], true).unwrap();

        assert_eq!(report.layer_reports[0].unknown_atom_count, 1);
        assert_eq!(report.layer_reports[0].seed_requests, 1);
        assert!(n.layers()[0].sigma().is_empty());
        let persisted = n.export_json();
        let restored =
            Auxein::<f32>::from_json(&persisted, Budget::units(n.budget_units()), "projection")
                .unwrap();
        assert_eq!(restored.export_json(), persisted);
    }

    #[test]
    fn zero_is_neither_learned_nor_emitted() {
        let mut n = make64();
        for _ in 0..5 {
            assert!(n.step(&[vec![0.0]], false).unwrap().readout.is_empty());
        }
        assert_eq!(n.layers()[0].cells().len(), 0);
        assert_eq!(n.layers()[0].sigma().len(), 0);
    }

    #[test]
    fn permutation_invariance() {
        let mut a = make64();
        let mut b = make64();
        let p1 = vec![vec![-2.0], vec![1.0], vec![4.0], vec![1.0]];
        let p2 = vec![vec![1.0], vec![4.0], vec![1.0], vec![-2.0]];
        for _ in 0..4 {
            a.step(&p1, false).unwrap();
            b.step(&p2, false).unwrap();
        }
        assert_eq!(a.export_json(), b.export_json());
    }

    #[test]
    fn external_mass_is_uniform_one() {
        let mut n = make64();
        let report = n.step(&[vec![-2.0], vec![2.0]], true).unwrap();
        assert_eq!(report.layer_reports[0].input_mass, 1.0);
        assert_eq!(n.layers()[0].sigma().len(), 2);
        assert_eq!(n.layers()[0].sigma()[0].weight(), n.beta() * 0.5);
        assert_eq!(n.layers()[0].sigma()[1].weight(), n.beta() * 0.5);
    }

    #[test]
    fn growth_is_all_or_nothing() {
        let mut small = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("1"), "auxein").unwrap();
        let report = small.step(&[vec![-2.0], vec![2.0]], false).unwrap();
        assert_eq!(small.layers()[0].sigma().len(), 0);
        assert!(report
            .transformations
            .iter()
            .any(|t| matches!(t, Transformation::GrowthReject { .. })));

        let mut roomy = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("2"), "auxein").unwrap();
        roomy.step(&[vec![-2.0], vec![2.0]], false).unwrap();
        assert_eq!(roomy.layers()[0].sigma().len(), 2);
    }

    #[test]
    fn context_frontier_is_in_growth_transaction() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut roomy = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let report = roomy.step(&[vec![1.0], vec![3.0]], true).unwrap();
        assert!(report.layer_reports[0].context_emitted);
        assert_eq!(roomy.layers().len(), 2);
        assert!(report.transformations.iter().any(|t| matches!(
            t,
            Transformation::GrowthCommit {
                layer_created: true,
                ..
            }
        )));
    }

    #[test]
    fn multiwinner_allocation_is_conservative() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":3.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let report = n.step(&[vec![3.0]], true).unwrap();
        let masses = &report.layer_reports[0].cell_responsibility_mass;
        assert_eq!(report.readout.len(), 2);
        assert!((stable_sum(masses.iter().copied()) - 1.0).abs() < 1e-15);
        assert!((masses[0] - 5.0 / 29.0).abs() < 1e-15);
        assert!((masses[1] - 24.0 / 29.0).abs() < 1e-15);
    }

    #[test]
    fn forced_solvency_destroys_work_then_knowledge() {
        let mut n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("20"), "auxein").unwrap();
        for x in [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0] {
            n.step(&[vec![x]], false).unwrap();
        }
        n.set_budget(Budget::kernels("0")).unwrap();
        let report = n.step(&[vec![0.0]], false).unwrap();
        assert_eq!(n.layers().len(), 1);
        assert!(n.layers()[0].cells().is_empty());
        assert!(n.layers()[0].sigma().is_empty());
        assert!(report
            .transformations
            .iter()
            .any(|t| matches!(t, Transformation::DestroyCells { .. })));
    }

    #[test]
    fn cell_value_ignores_support() {
        let a = Kernel::<f64>::new(1e-9, &[3.0], 1.0).unwrap();
        let b = Kernel::<f64>::new(1000.0, &[3.0], 1.0).unwrap();
        assert_eq!(Auxein::<f64>::cell_value(&a), Auxein::<f64>::cell_value(&b));
    }

    #[test]
    fn budget_is_not_serialized() {
        let n = make64();
        let state = n.export_json();
        assert!(!state.contains("budget_units"));
        assert!(!state.contains("\"budget\""));
    }

    #[test]
    fn clone_discards_transient_scratch_without_changing_behavior() {
        let mut a = make64();
        for _ in 0..3 {
            a.step(&[vec![2.0]], false).unwrap();
        }
        let mut b = a.clone();
        assert_eq!(a.export_json(), b.export_json());
        let ra = a.step(&[vec![2.0]], true).unwrap();
        let rb = b.step(&[vec![2.0]], true).unwrap();
        assert_eq!(ra, rb);
        assert_eq!(a.export_json(), b.export_json());
    }

    #[test]
    fn first_coordinate_window_is_exactly_safe() {
        let mut kernels = Vec::new();
        for i in 0..257 {
            let x = -64.0 + i as f64 * 0.5;
            let y = ((i * 37) % 29) as f64 * 0.125 - 1.75;
            let variance = ((i * 17) % 11) as f64 * 0.25;
            kernels.push(Kernel::<f64>::new(1.0 + i as f64 * 0.001, &[x, y], variance).unwrap());
        }
        sort_kernels(&mut kernels);

        for j in 0..400 {
            let x0 = -80.0 + j as f64 * 0.4;
            let x1 = ((j * 53) % 41) as f64 * 0.2 - 4.0;
            let point = [x0, x1];
            let d0 = norm2(&point);
            let range = first_coordinate_candidate_range(&kernels, x0, d0);
            for (i, kernel) in kernels.iter().enumerate() {
                if concern_scalar(kernel, &point, 0.0, d0).0 {
                    assert!(
                        range.contains(&i),
                        "concerned kernel {i} was excluded for point {point:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn strict_state_rejects_unrepresentable_f32_config() {
        let bad = r#"{"format_version":4,"dimension":1,"scalar":"f32","memory":10.1,"eta":0.7,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[]}]}"#;
        assert!(Auxein::<f32>::from_json(bad, Budget::kernels("10"), "x").is_err());
    }
    #[test]
    fn local_coalesce_matches_full_coalesce() {
        fn kernel(center: f64, variance: f64) -> Kernel<f64> {
            Kernel::new(1.0, &[center], variance).unwrap()
        }

        fn set_center(kernel: &mut Kernel<f64>, center: f64) {
            kernel.center[0] = center;
            kernel.norm2 = center * center;
            kernel.dirty = true;
        }

        fn check(mut kernels: Vec<Kernel<f64>>, touched: &[usize]) {
            for &index in touched {
                kernels[index].dirty = true;
            }
            let expected = coalesce_dirty(kernels.clone()).unwrap();
            let (actual, _) = coalesce_touched(kernels, touched, true).unwrap();
            assert_eq!(actual, expected);
        }

        let mut local = (0..8).map(|i| kernel(i as f64, 0.0)).collect::<Vec<_>>();
        set_center(&mut local[3], 3.25);
        check(local, &[3]);

        let mut crossing = (0..8).map(|i| kernel(i as f64, 0.0)).collect::<Vec<_>>();
        set_center(&mut crossing[3], 5.5);
        check(crossing, &[3]);

        let mut clone = (0..8).map(|i| kernel(i as f64, 0.0)).collect::<Vec<_>>();
        set_center(&mut clone[3], 4.0);
        check(clone, &[3]);

        let mut variance_crossing = vec![
            kernel(0.0, 0.0),
            kernel(1.0, 0.0),
            kernel(1.0, 1.0),
            kernel(2.0, 0.0),
        ];
        variance_crossing[1].variance = 2.0;
        variance_crossing[1].dirty = true;
        check(variance_crossing, &[1]);

        let mut adjacent = (0..8).map(|i| kernel(i as f64, 0.0)).collect::<Vec<_>>();
        set_center(&mut adjacent[3], 3.2);
        set_center(&mut adjacent[4], 4.2);
        check(adjacent, &[3, 4]);
    }

    #[test]
    fn ema_subnormal_support_does_not_square_the_denominator() {
        let mut kernels = vec![Kernel::<f64>::new(1e-200, &[0.0], 0.0).unwrap()];
        let presentation = [Atom {
            x: Arc::from(vec![1.0]),
            r: 1e-200,
            variance: 0.0,
            norm2: 1.0,
            zero: false,
        }];
        let mut targets = Targets::default();
        targets.reset(1, 1, true);
        targets.add_atom(0, 0, 1e-200);
        targets
            .apply_population(&mut kernels, &presentation, 0.5, 0.5)
            .unwrap();
        assert_eq!(kernels[0].weight, 1e-200);
        assert_eq!(kernels[0].center, vec![0.5]);
        assert_eq!(kernels[0].variance, 0.25);
        assert!(kernels[0].variance.is_finite());
    }

    #[test]
    fn single_atom_ema_matches_general_targets() {
        fn run<S: Scalar>() {
            let original = vec![
                Kernel::<S>::new(0.75, &[0.5, -1.0, 2.0], 0.25).unwrap(),
                Kernel::<S>::new(1.25, &[1.5, 0.25, -0.5], 0.5).unwrap(),
                Kernel::<S>::new(0.5, &[3.0, -2.0, 1.0], 0.125).unwrap(),
            ];
            let x = [1.0, -0.5, 0.75];
            let beta = 0.2;
            let lambda = 0.8;
            let responsibilities = [(0usize, 0.3), (2usize, 0.7)];

            let presentation = [Atom {
                x: Arc::from(x.to_vec()),
                r: 1.0,
                variance: 0.0,
                norm2: stable_sum(x.iter().map(|v| v * v)),
                zero: false,
            }];
            let mut general_targets = Targets::default();
            general_targets.reset(original.len(), x.len(), true);
            for &(index, weight) in &responsibilities {
                general_targets.add_atom(index, 0, weight);
            }
            let mut general = original.clone();
            general_targets
                .apply_population(&mut general, &presentation, beta, lambda)
                .unwrap();

            let mut single_targets = Targets::default();
            single_targets.reset(original.len(), x.len(), false);
            for &(index, weight) in &responsibilities {
                let distance2 =
                    stable_sum(original[index].center.iter().zip(x).map(|(&old, new)| {
                        let delta = old.to_f64() - new;
                        delta * delta
                    }));
                single_targets.mark_single(index, distance2);
                single_targets.set_single_weight(index, weight);
            }
            let mut single = original;
            single_targets
                .apply_single_population(&mut single, &x, beta, lambda)
                .unwrap();

            assert_eq!(single, general);
        }

        run::<f64>();
        run::<f32>();
    }
    #[test]
    fn internal_variance_participates_in_concern_bound() {
        let cell = Kernel::<f64>::new(1.0, &[2.0], 1.0).unwrap();
        assert!(concern_scalar(&cell, &[2.0], 0.5, 4.0).0);
        assert!(!concern_scalar(&cell, &[2.0], 6.0, 4.0).0);
        assert_eq!(concern_scalar(&cell, &[2.0], 0.5, 4.0).1, 4.0);
    }

    #[test]
    fn context_geometry_ignores_learning_responsibility() {
        let state_a = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let state_b = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":100.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut a = Auxein::<f64>::from_json(state_a, Budget::kernels("100"), "auxein").unwrap();
        let mut b = Auxein::<f64>::from_json(state_b, Budget::kernels("100"), "auxein").unwrap();
        let ra = a.step(&[vec![3.0]], true).unwrap();
        let rb = b.step(&[vec![3.0]], true).unwrap();
        assert_ne!(
            ra.layer_reports[0].cell_responsibility_mass,
            rb.layer_reports[0].cell_responsibility_mass
        );
        assert_eq!(
            ra.layer_reports[0].context_center,
            rb.layer_reports[0].context_center
        );
        assert_eq!(
            ra.layer_reports[0].context_variance,
            rb.layer_reports[0].context_variance
        );
        assert_eq!(
            ra.layer_reports[0].output_mass,
            rb.layer_reports[0].output_mass
        );
        assert_eq!(ra.layer_reports[0].context_center, Some(vec![1.5]));
        assert_eq!(ra.layer_reports[0].context_variance, Some(0.25));
    }

    #[test]
    fn context_mass_is_recognised_input_mass_without_duplication() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![1.0], vec![3.0], vec![-10.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.input_mass, 1.0);
        assert_eq!(layer.output_mass, 2.0 / 3.0);
        assert_eq!(layer.context_center, Some(vec![2.0]));
        assert_eq!(layer.context_variance, Some(1.0));

        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![3.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.output_mass, 1.0);
        assert_eq!(layer.context_center, Some(vec![1.5]));
        assert_eq!(layer.context_variance, Some(0.25));
    }

    #[test]
    fn singleton_and_zero_center_contexts_are_vertical_silence() {
        let singleton = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[2.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(singleton, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![2.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![2.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(0.0));
        assert!(!r.layer_reports[0].context_emitted);

        let symmetric = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-1.0],"V":0.0},{"W":1.0,"C":[1.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(symmetric, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![-1.0], vec![1.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![0.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(1.0));
        assert!(!r.layer_reports[0].context_emitted);
    }

    #[test]
    fn perfect_pair_emits_one_context_and_stops_after_l1_learns_it() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("1000"), "auxein").unwrap();
        let first = n.step(&[vec![1.0], vec![3.0]], true).unwrap();
        assert_eq!(first.layer_reports[0].context_center, Some(vec![2.0]));
        assert_eq!(first.layer_reports[0].context_variance, Some(1.0));
        assert!(first.layer_reports[0].context_emitted);
        assert_eq!(n.layers().len(), 2);
        n.step(&[vec![1.0], vec![3.0]], true).unwrap();
        n.step(&[vec![1.0], vec![3.0]], true).unwrap();
        assert_eq!(n.layers()[1].cells().len(), 1);
        let fourth = n.step(&[vec![1.0], vec![3.0]], true).unwrap();
        assert_eq!(n.layers().len(), 2);
        assert!(!fourth.layer_reports[1].context_emitted);
    }

    #[test]
    fn constant_input_with_two_explanations_does_not_build_deep_cascade() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":1.0},{"W":1.0,"C":[2.0],"V":1.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("1000"), "auxein").unwrap();
        for _ in 0..40 {
            n.step(&[vec![1.5]], false).unwrap();
        }
        assert!(n.layers().len() <= 2);
        if n.layers().len() == 2 {
            assert!(n.layers()[1].cells().len() <= 1);
        }
    }

    #[test]
    fn lazy_cell_decay_matches_eager_materialization() {
        fn run<S: Scalar>() {
            let scalar = S::NAME;
            let state = format!(
                "{{\"format_version\":4,\"dimension\":1,\"scalar\":\"{scalar}\",\"memory\":23.0,\"eta\":1.0,\"mode\":\"geometry\",\"steps_seen\":0,\"layers\":[{{\"sigma\":[],\"cells\":[{{\"W\":1.0,\"C\":[-3.0],\"V\":0.25}},{{\"W\":2.0,\"C\":[1.0],\"V\":0.25}},{{\"W\":3.0,\"C\":[4.0],\"V\":0.25}}]}}]}}"
            );
            let mut lazy = Auxein::<S>::from_json(&state, Budget::kernels("100"), "u").unwrap();
            let mut eager = lazy.clone();

            for step in 0..240 {
                if step == 70 {
                    lazy.set_eta(0.25).unwrap();
                    eager.set_eta(0.25).unwrap();
                } else if step == 130 {
                    lazy.set_eta(0.0).unwrap();
                    eager.set_eta(0.0).unwrap();
                } else if step == 160 {
                    lazy.set_eta(1.0).unwrap();
                    eager.set_eta(1.0).unwrap();
                }

                let x = match step % 17 {
                    0 => vec![vec![-3.0]],
                    1 => vec![vec![4.0]],
                    2 => vec![vec![1.0], vec![4.0]],
                    _ => vec![vec![1.0]],
                };
                let a = lazy.step(&x, step % 11 == 0).unwrap();
                let b = eager.step(&x, step % 11 == 0).unwrap();
                assert_eq!(a, b);

                for layer in &mut eager.layers {
                    let epoch = layer.cell_decay.epoch();
                    let lambda = layer.cell_decay.lambda();
                    for cell in &mut layer.cells {
                        cell.materialize_weight_at(epoch, lambda);
                    }
                }
                assert_eq!(lazy.export_json(), eager.export_json());
            }
        }
        run::<f64>();
        run::<f32>();
    }

    #[test]
    fn lazy_decay_clock_is_clone_local() {
        let state = r#"{"format_version":4,"dimension":1,"scalar":"f64","memory":31.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-3.0],"V":0.25},{"W":2.0,"C":[1.0],"V":0.25},{"W":3.0,"C":[4.0],"V":0.25}]}]}"#;
        let mut original = Auxein::<f64>::from_json(state, Budget::kernels("100"), "u").unwrap();
        for _ in 0..80 {
            original.step(&[vec![1.0]], false).unwrap();
        }
        let before = original.export_json();
        let mut cloned = original.clone();
        for _ in 0..20 {
            cloned.step(&[vec![4.0]], false).unwrap();
        }
        assert_eq!(original.export_json(), before);
        assert_ne!(cloned.export_json(), before);
    }

    #[test]
    fn temporal_packing() {
        let n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            1.0,
            Mode::Temporal,
            Budget::kernels("0"),
            "auxein",
        )
        .unwrap();
        assert_eq!(n.temporal_kernel_units().unwrap(), 32);
        assert_eq!(n.layer_units().unwrap(), 57);
        assert_eq!(n.min_units().unwrap(), 107);
        assert_eq!(n.maintenance_units().unwrap(), 107);
    }

    #[test]
    fn mode_is_geometry_by_default_and_strict() {
        let n = make64();
        assert_eq!(n.mode(), Mode::Geometry);
        assert!(n.export_json().contains("\"mode\":\"geometry\""));
        assert_eq!(Mode::parse("geometry").unwrap(), Mode::Geometry);
        assert_eq!(Mode::parse("temporal").unwrap(), Mode::Temporal);
        assert_eq!(Mode::parse("predictive").unwrap(), Mode::Predictive);
        assert!(Mode::parse("future").is_err());
    }

    #[test]
    fn predictive_projects_existing_temporal_cell_from_current_context() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Predictive,
            Budget::kernels("100"),
            "lab",
        )
        .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        // Full temporal variance is deliberately huge: prediction projects only
        // the source/target centers and cannot reconstruct endpoint variance.
        n.layers[0].temporal_cells = vec![Kernel::new(7.0, &[1.0, 5.0], 123.0).unwrap()];

        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.concepts().len(), 1);
        assert!(report.readout.sequences().is_empty());
        assert_eq!(report.readout.predictions().len(), 1);
        let prediction = &report.readout.predictions()[0];
        assert_eq!(prediction.universe.as_ref(), "lab");
        assert_eq!(prediction.current_context, vec![1.0]);
        assert_eq!(prediction.recognised_source, vec![1.0]);
        assert_eq!(prediction.predicted_successor, vec![5.0]);
    }

    #[test]
    fn predictive_branching_emits_every_known_successor_without_selection() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[1.0, 3.0], 0.0).unwrap(),
            Kernel::new(1.0, &[1.0, 5.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0, 9.0], 0.0).unwrap(), // must not chain 1->3->9
        ];

        let report = n.step(&[vec![1.0]], false).unwrap();
        let successors: Vec<Vec<f64>> = report
            .readout
            .predictions()
            .iter()
            .map(|p| p.predicted_successor.clone())
            .collect();
        assert_eq!(successors, vec![vec![3.0], vec![5.0]]);
    }

    #[test]
    fn predictive_zero_source_is_silent_and_zero_target_is_explicit() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[0.0, 9.0], 0.0).unwrap(),
            Kernel::new(1.0, &[1.0, 0.0], 0.0).unwrap(),
        ];

        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.predictions().len(), 1);
        assert_eq!(report.readout.predictions()[0].recognised_source, vec![1.0]);
        assert_eq!(
            report.readout.predictions()[0].predicted_successor,
            vec![0.0]
        );
    }

    #[test]
    fn newly_promoted_temporal_cell_predicts_only_from_next_step() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];

        let first = n.step(&[vec![1.0]], false).unwrap();
        assert!(first.readout.predictions().is_empty());
        let second = n.step(&[vec![1.0]], false).unwrap();
        assert!(second.readout.predictions().is_empty());
        assert_eq!(n.layers[0].temporal_sigma.len(), 1);
        let third = n.step(&[vec![1.0]], false).unwrap();
        assert!(third.readout.predictions().is_empty());
        assert_eq!(n.layers[0].temporal_cells.len(), 1);
        let fourth = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(fourth.readout.predictions().len(), 1);
        assert_eq!(
            fourth.readout.predictions()[0].predicted_successor,
            vec![1.0]
        );
    }

    #[test]
    fn predictive_and_temporal_have_identical_persistent_trajectory_and_cost() {
        let mut temporal = make_temporal64();
        let mut predictive = make_predictive64();
        let stream = [1.0, 3.0, 1.0, 3.0, 9.0, 1.0, 3.0, 1.0, 3.0];

        assert_eq!(
            temporal.maintenance_units().unwrap(),
            predictive.maintenance_units().unwrap()
        );
        for x in stream {
            temporal.step(&[vec![x]], false).unwrap();
            predictive.step(&[vec![x]], false).unwrap();
            assert_eq!(temporal.layers, predictive.layers);
            assert_eq!(temporal.steps_seen, predictive.steps_seen);
            assert_eq!(
                temporal.maintenance_units().unwrap(),
                predictive.maintenance_units().unwrap()
            );
        }
    }

    #[test]
    fn predictive_scale_and_signed_orthogonal_invariance() {
        fn build(
            current: &[f64],
            source: &[f64],
            target: &[f64],
            temporal_variance: f64,
        ) -> Auxein<f64> {
            let mut n = Auxein::<f64>::new_with_mode(
                2,
                10.0,
                0.0,
                Mode::Predictive,
                Budget::kernels("100"),
                "u",
            )
            .unwrap();
            n.layers[0].cells = vec![Kernel::new(1.0, current, 0.0).unwrap()];
            let mut temporal = source.to_vec();
            temporal.extend_from_slice(target);
            n.layers[0].temporal_cells =
                vec![Kernel::new(1.0, &temporal, temporal_variance).unwrap()];
            n
        }

        let mut base = build(&[1.0, 0.0], &[0.9, 0.1], &[0.0, 2.0], 77.0);
        let base_report = base.step(&[vec![1.0, 0.0]], false).unwrap();
        let p = &base_report.readout.predictions()[0];

        let mut scaled = build(&[10.0, 0.0], &[9.0, 1.0], &[0.0, 20.0], 7700.0);
        let scaled_report = scaled.step(&[vec![10.0, 0.0]], false).unwrap();
        let ps = &scaled_report.readout.predictions()[0];
        assert_eq!(ps.universe, p.universe);
        assert_eq!(
            ps.current_context,
            p.current_context
                .iter()
                .map(|x| 10.0 * x)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ps.recognised_source,
            p.recognised_source
                .iter()
                .map(|x| 10.0 * x)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ps.predicted_successor,
            p.predicted_successor
                .iter()
                .map(|x| 10.0 * x)
                .collect::<Vec<_>>()
        );

        // Q(x,y)=(-y,x), a signed orthogonal permutation.
        let mut rotated = build(&[0.0, 1.0], &[-0.1, 0.9], &[-2.0, 0.0], 77.0);
        let rotated_report = rotated.step(&[vec![0.0, 1.0]], false).unwrap();
        let pr = &rotated_report.readout.predictions()[0];
        let q = |v: &[f64]| vec![-v[1], v[0]];
        assert_eq!(pr.universe, p.universe);
        assert_eq!(pr.current_context, q(&p.current_context));
        assert_eq!(pr.recognised_source, q(&p.recognised_source));
        assert_eq!(pr.predicted_successor, q(&p.predicted_successor));
    }

    #[test]
    fn predictive_roundtrip_preserves_mode_state_and_readout() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Predictive,
            Budget::kernels("100"),
            "roundtrip",
        )
        .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 5.0], 0.0).unwrap()];
        let state = n.export_json();
        assert!(state.contains("\"format_version\":4"));
        assert!(state.contains("\"mode\":\"predictive\""));

        let mut restored =
            Auxein::<f64>::from_json(&state, Budget::units(n.budget_units()), "roundtrip").unwrap();
        assert_eq!(restored.mode(), Mode::Predictive);
        assert_eq!(restored.export_json(), state);
        let report = restored.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.predictions().len(), 1);
        assert_eq!(
            report.readout.predictions()[0].predicted_successor,
            vec![5.0]
        );
    }

    #[test]
    fn temporal_readout_recognises_adjacent_order() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "lab",
        )
        .unwrap();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[5.0], 0.0).unwrap(),
        ];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 5.0], 0.0).unwrap()];

        let first = n.step(&[vec![1.0]], false).unwrap();
        assert!(first.readout.sequences().is_empty());
        let second = n.step(&[vec![5.0]], false).unwrap();
        assert_eq!(second.readout.concepts().len(), 1);
        assert_eq!(second.readout.concepts()[0].local_input.as_ref(), &[5.0]);
        assert_eq!(second.readout.concepts()[0].recognised, vec![5.0]);
        assert_eq!(second.readout.sequences().len(), 1);
        let seq = &second.readout.sequences()[0];
        assert_eq!(seq.universe.as_ref(), "lab");
        assert_eq!(seq.previous_input.as_ref(), &[1.0]);
        assert_eq!(seq.current_input.as_ref(), &[5.0]);
        assert_eq!(seq.previous_recognised, vec![1.0]);
        assert_eq!(seq.current_recognised, vec![5.0]);
        let third = n.step(&[vec![1.0]], false).unwrap();
        assert!(third.readout.sequences().is_empty());
    }

    #[test]
    fn temporal_recurrence_promotes_only_after_recurrence() {
        let mut n = make_temporal64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[5.0], 0.0).unwrap(),
        ];
        n.step(&[vec![1.0]], false).unwrap();
        n.step(&[vec![5.0]], false).unwrap();
        assert_eq!(n.layers[0].temporal_sigma.len(), 1);
        assert_eq!(n.layers[0].temporal_sigma[0].center(), vec![1.0, 5.0]);
        n.step(&[vec![1.0]], false).unwrap();
        n.step(&[vec![5.0]], false).unwrap();
        assert!(n.layers[0]
            .temporal_cells
            .iter()
            .any(|cell| cell.center() == vec![1.0, 5.0]));
    }

    #[test]
    fn missing_context_breaks_temporal_chain() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "auxein",
        )
        .unwrap();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0], 0.0).unwrap(),
        ];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 3.0], 0.0).unwrap()];
        n.step(&[vec![1.0]], false).unwrap();
        let gap = n.step(&[vec![99.0]], false).unwrap();
        assert!(gap.readout.concepts().is_empty());
        assert!(!n.summary().unwrap().previous_context_per_layer[0]);
        let after = n.step(&[vec![3.0]], false).unwrap();
        assert!(after.readout.sequences().is_empty());
    }

    #[test]
    fn eta_zero_freezes_temporal_learning_but_previous_advances() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "auxein",
        )
        .unwrap();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0], 0.0).unwrap(),
        ];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 3.0], 1.0).unwrap()];
        let before = n.layers[0].temporal_cells.clone();
        n.step(&[vec![1.0]], false).unwrap();
        let report = n.step(&[vec![3.0]], false).unwrap();
        assert!(!report.readout.sequences().is_empty());
        assert_eq!(n.layers[0].temporal_cells, before);
        assert_eq!(n.layers[0].previous.as_ref().unwrap().center(), vec![3.0]);
    }

    #[test]
    fn temporal_roundtrip_preserves_previous_context() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "roundtrip",
        )
        .unwrap();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0], 0.0).unwrap(),
        ];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 3.0], 0.0).unwrap()];
        n.step(&[vec![1.0]], false).unwrap();
        let state = n.export_json();
        let mut restored =
            Auxein::<f64>::from_json(&state, Budget::units(n.budget_units()), "roundtrip").unwrap();
        assert_eq!(restored.export_json(), state);
        let report = restored.step(&[vec![3.0]], false).unwrap();
        assert_eq!(report.readout.sequences().len(), 1);
        let seq = &report.readout.sequences()[0];
        assert_eq!(seq.previous_input.as_ref(), &[1.0]);
        assert_eq!(seq.current_input.as_ref(), &[3.0]);
        assert_eq!(seq.previous_recognised, vec![1.0]);
        assert_eq!(seq.current_recognised, vec![3.0]);
    }

    #[test]
    fn temporal_growth_shares_one_global_transaction() {
        let mut n = make_temporal64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0], 0.0).unwrap(),
        ];
        n.step(&[vec![1.0]], false).unwrap();
        let base = n.maintenance_units().unwrap();
        let extra = n
            .kernel_units()
            .unwrap()
            .max(n.temporal_kernel_units().unwrap());
        n.set_budget(Budget::units(base + extra)).unwrap();
        let report = n.step(&[vec![3.0], vec![9.0]], false).unwrap();
        let growth = report
            .transformations
            .iter()
            .rev()
            .find(|t| matches!(t, Transformation::GrowthReject { .. }))
            .expect("growth rejection");
        match growth {
            Transformation::GrowthReject {
                geometric_seeds,
                temporal_seeds,
                ..
            } => {
                assert_eq!(*geometric_seeds, 1);
                assert_eq!(*temporal_seeds, 1);
            }
            _ => unreachable!(),
        }
        assert!(n.layers[0].sigma.is_empty());
        assert!(n.layers[0].temporal_sigma.is_empty());
    }

    #[test]
    fn forced_contraction_invalidates_temporal_previous() {
        let mut n = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            0.0,
            Mode::Temporal,
            Budget::kernels("100"),
            "auxein",
        )
        .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 1.0], 0.0).unwrap()];
        n.step(&[vec![1.0]], false).unwrap();
        assert!(n.layers[0].previous.is_some());
        n.set_budget(Budget::units(n.min_units().unwrap())).unwrap();
        n.step(&[vec![1.0]], false).unwrap();
        assert!(n.layers[0].cells.is_empty());
        assert!(n.layers[0].temporal_cells.is_empty());
        assert!(n.layers[0].previous.is_none());
    }

    #[test]
    fn temporal_product_kernel_is_exact_direct_sum_quotient() {
        let n = make_temporal64();
        let previous = Kernel64 {
            weight: 0.5,
            center: vec![2.0],
            variance: 1.0,
        };
        let current = Kernel64 {
            weight: 0.25,
            center: vec![7.0],
            variance: 4.0,
        };
        let temporal = n.temporal_atom(&previous, &current);
        assert_eq!(temporal.r, 0.125);
        assert_eq!(temporal.x.as_ref(), &[2.0, 7.0]);
        assert_eq!(temporal.variance, 5.0);
    }

    #[test]
    fn temporal_population_does_not_age_without_temporal_presentation() {
        let mut n = make_temporal64();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(2.0, &[1.0, 1.0], 0.5).unwrap()];
        let before = n.layers[0].temporal_cells[0].clone();
        n.step(&[vec![99.0]], false).unwrap();
        assert_eq!(n.layers[0].temporal_cells[0], before);
    }

    #[test]
    fn temporal_mode_preserves_geometric_trajectory_with_sufficient_budget() {
        let mut g = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("10000"), "auxein").unwrap();
        let mut t = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            1.0,
            Mode::Temporal,
            Budget::kernels("10000"),
            "auxein",
        )
        .unwrap();
        let sequence = [
            vec![vec![1.0]],
            vec![vec![5.0]],
            vec![vec![1.0]],
            vec![vec![5.0]],
            vec![vec![1.0], vec![5.0]],
        ];
        for _ in 0..5 {
            for presentation in &sequence {
                g.step(presentation, false).unwrap();
                t.step(presentation, false).unwrap();
            }
        }
        assert_eq!(g.layers.len(), t.layers.len());
        for (gl, tl) in g.layers.iter().zip(&t.layers) {
            assert_eq!(gl.cells, tl.cells);
            assert_eq!(gl.sigma, tl.sigma);
        }
    }

    #[test]
    fn temporal_scale_invariance() {
        let mut a = Auxein::<f64>::new_with_mode(
            1,
            10.0,
            1.0,
            Mode::Temporal,
            Budget::kernels("10000"),
            "auxein",
        )
        .unwrap();
        let mut b = a.clone();
        a.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[5.0], 0.0).unwrap(),
        ];
        b.layers[0].cells = vec![
            Kernel::new(1.0, &[10.0], 0.0).unwrap(),
            Kernel::new(1.0, &[50.0], 0.0).unwrap(),
        ];
        let sequence = [
            vec![vec![1.0]],
            vec![vec![5.0]],
            vec![vec![1.0]],
            vec![vec![5.0]],
        ];
        for _ in 0..5 {
            for presentation in &sequence {
                a.step(presentation, false).unwrap();
                let scaled = presentation
                    .iter()
                    .map(|v| vec![10.0 * v[0]])
                    .collect::<Vec<_>>();
                b.step(&scaled, false).unwrap();
            }
        }
        assert_eq!(
            a.layers[0].temporal_cells.len(),
            b.layers[0].temporal_cells.len()
        );
        for (ka, kb) in a.layers[0]
            .temporal_cells
            .iter()
            .zip(&b.layers[0].temporal_cells)
        {
            assert_eq!(
                ka.center()
                    .into_iter()
                    .map(|x| 10.0 * x)
                    .collect::<Vec<_>>(),
                kb.center()
            );
            assert!((100.0 * ka.variance() - kb.variance()).abs() < 1e-12);
            assert!((ka.weight() - kb.weight()).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_to_zero_is_temporally_silent() {
        let mut n = make_temporal64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[-1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
        ];
        n.step(&[vec![-1.0], vec![1.0]], false).unwrap();
        n.step(&[vec![-1.0], vec![1.0]], false).unwrap();
        assert!(n.layers[0].temporal_sigma.is_empty());
        assert!(n.layers[0].temporal_cells.is_empty());
    }
}
