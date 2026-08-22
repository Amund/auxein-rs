#![forbid(unsafe_code)]

//! Auxein v0.5.0 production core.
//!
//! The crate is deliberately dependency-free. Persistent geometry is stored
//! in the selected scalar (`f32` or `f64`), every cognitive calculation is
//! performed in `f64`, and material accounting is exact integer arithmetic.

mod decimal;
mod json;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::io::Write as IoWrite;
use std::mem;
use std::sync::{
    atomic::{AtomicU64, Ordering as AtomicOrdering},
    Arc,
};

pub const FORMAT_VERSION: u64 = 5;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Geometry,
    Predictive,
}

impl Mode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "geometry" => Ok(Self::Geometry),
            "predictive" => Ok(Self::Predictive),
            _ => Err(Error::Invalid(
                "mode must be 'geometry' or 'predictive'".into(),
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Predictive => "predictive",
        }
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
    fn max_finite() -> Self;
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

    fn max_finite() -> Self {
        f32::MAX
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

    fn max_finite() -> Self {
        f64::MAX
    }
}

// CELL support decays homothetically and has no authority while the CELL is
// outside the concern set.  The layer clock lets untouched CELLs defer those
// exact scalar projections until their support is observed again.  Changing
// eta materializes every pending decay before installing a new clock, so a
// kernel never spans two lambda values.
#[derive(Debug)]
struct DecayClock {
    epoch: AtomicU64,
    lambda_bits: AtomicU64,
}

impl DecayClock {
    fn new(epoch: u64, lambda: f64) -> Self {
        Self {
            epoch: AtomicU64::new(epoch),
            lambda_bits: AtomicU64::new(lambda.to_bits()),
        }
    }

    #[inline]
    fn epoch(&self) -> u64 {
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
    decay_epoch: u64,
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
    fn materialize_weight_at(&mut self, epoch: u64, lambda: f64) -> f64 {
        if self.decay_epoch != epoch {
            self.weight = decayed_weight::<S>(self.weight, self.decay_epoch, epoch, lambda);
            self.decay_epoch = epoch;
        }
        self.weight.to_f64()
    }

    fn bind_decay_clock(&mut self, clock: Arc<DecayClock>, epoch: u64) {
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
        let mut sigma = self.sigma.clone();
        for kernel in &mut sigma {
            kernel.bind_decay_clock(clock.clone(), kernel.decay_epoch);
        }
        let mut cells = self.cells.clone();
        for cell in &mut cells {
            cell.bind_decay_clock(clock.clone(), cell.decay_epoch);
        }
        let temporal_epoch = self.temporal_decay.epoch();
        let temporal_clock = Arc::new(DecayClock::new(
            temporal_epoch,
            self.temporal_decay.lambda(),
        ));
        let mut temporal_sigma = self.temporal_sigma.clone();
        for kernel in &mut temporal_sigma {
            kernel.bind_decay_clock(temporal_clock.clone(), kernel.decay_epoch);
        }
        let mut temporal_cells = self.temporal_cells.clone();
        for cell in &mut temporal_cells {
            cell.bind_decay_clock(temporal_clock.clone(), cell.decay_epoch);
        }
        Self {
            sigma,
            cells,
            cell_decay: clock,
            temporal_sigma,
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
pub struct InputAtom {
    pub weight: f64,
    pub center: Vec<f64>,
    pub variance: f64,
}

impl InputAtom {
    pub fn new(weight: f64, center: Vec<f64>, variance: f64) -> Self {
        Self {
            weight,
            center,
            variance,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputAtom {
    pub weight: f64,
    pub center: Vec<f64>,
    pub variance: f64,
}

pub type OutputPresentation = Vec<OutputAtom>;

#[derive(Clone, Debug, PartialEq)]
pub enum Readout {
    Geometry {
        present: Vec<OutputPresentation>,
    },
    Predictive {
        present: Vec<OutputPresentation>,
        future: Vec<OutputPresentation>,
    },
}

impl Readout {
    pub fn present(&self) -> &[OutputPresentation] {
        match self {
            Self::Geometry { present } | Self::Predictive { present, .. } => present,
        }
    }

    pub fn future(&self) -> &[OutputPresentation] {
        match self {
            Self::Geometry { .. } => &[],
            Self::Predictive { future, .. } => future,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.present().is_empty() && self.future().is_empty()
    }

    pub fn len(&self) -> usize {
        self.present().len()
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
    pub knowledge_mass: f64,
    pub present_atom_count: usize,
    pub present_mass: f64,
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
    sequence_open: bool,
}

impl<S: Scalar> Clone for Auxein<S> {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            memory: self.memory,
            eta: self.eta,
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
            sequence_open: false,
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
    fn reset(&mut self, count: usize, dimension: usize, need_centers: bool) -> Result<()> {
        for index in self.touched.drain(..) {
            let Some(weight) = self.weights.get_mut(index) else {
                return Err(Error::Inexecutable(
                    "scratch index escaped its population".into(),
                ));
            };
            *weight = 0.0;
            if let Some(batch) = self.batches.get_mut(index) {
                batch.clear();
            }
        }
        self.dimension = dimension;
        if self.weights.len() < count {
            self.weights
                .try_reserve(count - self.weights.len())
                .map_err(|_| Error::Inexecutable("cannot reserve target weights".into()))?;
            self.variances
                .try_reserve(count - self.variances.len())
                .map_err(|_| Error::Inexecutable("cannot reserve target variances".into()))?;
            self.weights.resize(count, 0.0);
            self.variances.resize(count, 0.0);
        }
        if self.batches.len() < count {
            self.batches
                .try_reserve(count - self.batches.len())
                .map_err(|_| Error::Inexecutable("cannot reserve target batches".into()))?;
            self.batches.resize_with(count, Vec::new);
        }
        if need_centers {
            let center_count = count
                .checked_mul(dimension)
                .ok_or_else(|| Error::Inexecutable("target scratch dimension overflow".into()))?;
            if self.centers.len() < center_count {
                self.centers
                    .try_reserve(center_count - self.centers.len())
                    .map_err(|_| Error::Inexecutable("cannot reserve target centers".into()))?;
                self.centers.resize(center_count, 0.0);
            }
        }
        Ok(())
    }

    fn capacity_bytes(&self) -> usize {
        let mut total = 0usize;
        let mut add = |capacity: usize, size: usize| {
            total = total.saturating_add(capacity.saturating_mul(size));
        };
        add(self.weights.capacity(), std::mem::size_of::<f64>());
        add(self.centers.capacity(), std::mem::size_of::<f64>());
        add(self.variances.capacity(), std::mem::size_of::<f64>());
        add(self.touched.capacity(), std::mem::size_of::<usize>());
        add(self.changed.capacity(), std::mem::size_of::<usize>());
        add(
            self.batches.capacity(),
            std::mem::size_of::<Vec<TargetContribution>>(),
        );
        for batch in &self.batches {
            add(batch.capacity(), std::mem::size_of::<TargetContribution>());
        }
        add(self.sum_terms.capacity(), std::mem::size_of::<f64>());
        add(self.variance_terms.capacity(), std::mem::size_of::<f64>());
        add(self.sum_partials.capacity(), std::mem::size_of::<f64>());
        total
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

    #[cfg(test)]
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

    #[cfg(test)]
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
        epoch: u64,
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
        epoch: u64,
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
        if !(a >= 0.0 && b >= 0.0) || (a == 0.0 && b == 0.0) {
            return Err(Error::Invalid("EMA produced nonpositive support".into()));
        }
        // Compute the convex coefficient homogeneously if `a + b` overflows.
        // The coefficient, not the particular intermediate sum, carries the
        // cognitive authority.
        let ratio = if total.is_finite() && total > 0.0 {
            b / total
        } else {
            let scale = a.max(b);
            if !(scale > 0.0 && scale.is_finite()) {
                return Err(Error::Invalid("EMA produced invalid support".into()));
            }
            let an = a / scale;
            let bn = b / scale;
            bn / (an + bn)
        };
        let old_ratio = 1.0 - ratio;
        let mut changed = false;
        for (old_component, &new) in old.center.iter_mut().zip(target_center) {
            let old_value = old_component.to_f64();
            // Convex form avoids overflowing `new - old` for opposite extreme
            // finite coordinates.  A convex combination of representable
            // endpoints is itself representable.
            let value = structural_zero(old_ratio * old_value + ratio * new);
            let projected = S::from_f64(value)?;
            changed |= projected != *old_component;
            *old_component = projected;
        }
        let variance_raw = old_ratio * old.variance.to_f64()
            + ratio * target_variance
            + old_ratio * ratio * delta2;
        if variance_raw.is_nan() || variance_raw < 0.0 {
            return Err(Error::Invalid("EMA produced invalid variance".into()));
        }
        let scalar_max = S::max_finite().to_f64();
        let variance_value = structural_zero(if variance_raw.is_finite() {
            variance_raw.min(scalar_max)
        } else {
            scalar_max
        });
        let weight_value = if total.is_finite() {
            total.min(scalar_max)
        } else {
            scalar_max
        };
        let mut weight = S::from_f64(weight_value)?;
        if weight.to_f64() <= 0.0 {
            weight = S::min_positive();
        }
        let variance = S::from_f64(variance_value)?;
        changed |= variance != old.variance;
        old.weight = weight;
        old.variance = variance;
        old.norm2 = norm2_scalar(&old.center);
        old.dirty = changed;
        Ok(changed)
    }
}

#[inline]
fn decay_power(mut base: f64, mut exponent: u64) -> f64 {
    let mut factor = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 {
            factor *= base;
            if factor == 0.0 {
                return 0.0;
            }
        }
        exponent >>= 1;
        if exponent != 0 {
            base *= base;
            if base == 0.0 {
                return 0.0;
            }
        }
    }
    factor
}

#[inline]
fn decayed_weight<S: Scalar>(weight: S, from_epoch: u64, to_epoch: u64, lambda: f64) -> S {
    debug_assert!(from_epoch <= to_epoch);
    let age = to_epoch - from_epoch;
    if age == 0 || lambda >= 1.0 {
        return weight;
    }
    if lambda <= 0.0 {
        return S::min_positive();
    }

    let initial = weight.to_f64();
    let factor = decay_power(lambda, age);
    let mut value = initial * factor;
    if value == 0.0 && initial > 0.0 {
        // Binary exponentiation can underflow the standalone factor even when
        // initial * lambda^age remains representable.  Recover the product in
        // log2 space; this is an implementation choice, not cognitive state.
        let exponent = initial.log2() + (age as f64) * lambda.log2();
        value = exponent.exp2();
    }
    let mut projected = <S as sealed::Sealed>::from_finite(value);
    if projected.to_f64() <= 0.0 {
        projected = S::min_positive();
    }
    projected
}

#[cfg(test)]
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
    knowledge: Vec<OutputAtom>,
    seed_requests: Vec<Kernel64>,
    transformations: Vec<Transformation>,
    report: Option<LayerReport>,
}

impl<S: Scalar> Auxein<S> {
    pub fn new(dimension: usize, memory: f64, eta: f64, budget: Budget) -> Result<Self> {
        Self::new_with_mode(dimension, memory, eta, Mode::Geometry, budget)
    }

    pub fn new_with_mode(
        dimension: usize,
        memory: f64,
        eta: f64,
        mode: Mode,
        budget: Budget,
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
        let memory = S::from_f64(memory)?;
        let eta = S::from_f64(eta)?;
        let mut out = Self {
            dimension,
            memory,
            eta,
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
            sequence_open: false,
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

    /// Capacity retained only for transient per-step scratch buffers.
    pub fn transient_memory_capacity_bytes(&self) -> usize {
        self.scratch_targets
            .capacity_bytes()
            .saturating_add(
                self.scratch_concerned
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(usize, f64)>()),
            )
            .saturating_add(
                self.scratch_unknown
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }

    /// Drop transient high-water allocations without touching persistent
    /// cognition. Useful after exceptional bursts in long-running hosts.
    pub fn compact_transient_memory(&mut self) {
        self.scratch_targets = Targets::default();
        self.scratch_concerned = Vec::new();
        self.scratch_unknown = Vec::new();
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
            Mode::Predictive => 33u64
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
            for kernel in &mut layer.sigma {
                kernel.materialize_weight_at(epoch, lambda);
            }
            for cell in &mut layer.cells {
                cell.materialize_weight_at(epoch, lambda);
            }
            if self.mode.predictive() {
                let epoch = layer.temporal_decay.epoch();
                let lambda = layer.temporal_decay.lambda();
                for kernel in &mut layer.temporal_sigma {
                    kernel.materialize_weight_at(epoch, lambda);
                }
                for cell in &mut layer.temporal_cells {
                    cell.materialize_weight_at(epoch, lambda);
                }
            }
        }
        self.eta = eta;
        self.refresh_clock();
        for layer in &mut self.layers {
            for kernel in &mut layer.sigma {
                kernel.decay_epoch = 0;
            }
            for cell in &mut layer.cells {
                cell.decay_epoch = 0;
            }
            layer.cell_decay.reset(self.lambda);
            if self.mode.predictive() {
                for kernel in &mut layer.temporal_sigma {
                    kernel.decay_epoch = 0;
                }
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
            if self.mode.predictive() {
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
        let variance = cell.variance.to_f64();
        if variance <= 0.0 {
            return 1.0;
        }
        let norm2 = cell.norm2;
        let denominator = norm2 + variance;
        if norm2.is_finite() && norm2 > 0.0 && denominator.is_finite() {
            return norm2 / denominator;
        }

        // K is homogeneous.  Re-evaluate it in scaled coordinates whenever a
        // squared norm overflowed/underflowed.  A persistent CELL center is
        // nonzero, so K must stay strictly positive.
        let mut scale = variance.sqrt();
        for &component in &cell.center {
            scale = scale.max(component.to_f64().abs());
        }
        if !(scale > 0.0 && scale.is_finite()) {
            return f64::MIN_POSITIVE;
        }
        let center2 = stable_sum(cell.center.iter().map(|&component| {
            let x = component.to_f64() / scale;
            x * x
        }));
        let v = {
            let x = variance.sqrt() / scale;
            x * x
        };
        let denominator = center2 + v;
        let k = if denominator > 0.0 && denominator.is_finite() {
            center2 / denominator
        } else {
            0.0
        };
        if k > 0.0 && k.is_finite() {
            k.min(1.0)
        } else {
            f64::MIN_POSITIVE
        }
    }

    /// Process one uniform vector presentation as an atomic sequence.
    pub fn step(&mut self, presentation: &[Vec<f64>], detailed_report: bool) -> Result<StepReport> {
        if self.sequence_open {
            return Err(Error::Invalid(
                "step cannot be used while a sequence is open".into(),
            ));
        }
        let atoms = self.presentation(presentation)?;
        self.invalidate_previous();
        self.sequence_open = true;
        let result = self.step_atoms(atoms, detailed_report);
        self.invalidate_previous();
        self.sequence_open = false;
        result
    }

    /// Process one canonical weighted presentation as an atomic sequence.
    pub fn step_weighted(
        &mut self,
        presentation: &[InputAtom],
        detailed_report: bool,
    ) -> Result<StepReport> {
        if self.sequence_open {
            return Err(Error::Invalid(
                "step_weighted cannot be used while a sequence is open".into(),
            ));
        }
        let atoms = self.weighted_presentation(presentation)?;
        self.invalidate_previous();
        self.sequence_open = true;
        let result = self.step_atoms(atoms, detailed_report);
        self.invalidate_previous();
        self.sequence_open = false;
        result
    }

    /// Open an explicit causal sequence. By default restored causal registers
    /// are discarded; `resume=true` is the explicit opt-in for mid-sequence
    /// continuation after state restoration.
    pub fn begin_sequence(&mut self, resume: bool) -> Result<()> {
        if self.sequence_open {
            return Err(Error::Invalid("a sequence is already open".into()));
        }
        if !resume {
            self.invalidate_previous();
        }
        self.sequence_open = true;
        Ok(())
    }

    /// Close the current sequence and destroy all causal continuity.
    pub fn end_sequence(&mut self) -> Result<()> {
        if !self.sequence_open {
            return Err(Error::Invalid("no sequence is open".into()));
        }
        self.invalidate_previous();
        self.sequence_open = false;
        Ok(())
    }

    /// Process one uniform vector presentation inside an explicitly open sequence.
    pub fn sequence_step(
        &mut self,
        presentation: &[Vec<f64>],
        detailed_report: bool,
    ) -> Result<StepReport> {
        if !self.sequence_open {
            return Err(Error::Invalid(
                "sequence_step requires begin_sequence".into(),
            ));
        }
        let atoms = self.presentation(presentation)?;
        self.step_atoms(atoms, detailed_report)
    }

    /// Process one weighted presentation inside an explicitly open sequence.
    pub fn sequence_step_weighted(
        &mut self,
        presentation: &[InputAtom],
        detailed_report: bool,
    ) -> Result<StepReport> {
        if !self.sequence_open {
            return Err(Error::Invalid(
                "sequence_step_weighted requires begin_sequence".into(),
            ));
        }
        let atoms = self.weighted_presentation(presentation)?;
        self.step_atoms(atoms, detailed_report)
    }

    /// Process a finite nonempty explicit sequence of uniform vector presentations.
    pub fn sequence(
        &mut self,
        presentations: &[Vec<Vec<f64>>],
        detailed_report: bool,
    ) -> Result<Vec<StepReport>> {
        if presentations.is_empty() {
            return Err(Error::Invalid("sequence must be nonempty".into()));
        }
        self.begin_sequence(false)?;
        let mut reports = Vec::new();
        reports
            .try_reserve(presentations.len())
            .map_err(|_| Error::Inexecutable("cannot reserve sequence reports".into()))?;
        let mut failure = None;
        for presentation in presentations {
            match self.sequence_step(presentation, detailed_report) {
                Ok(report) => reports.push(report),
                Err(err) => {
                    failure = Some(err);
                    break;
                }
            }
        }
        let close = self.end_sequence();
        if let Some(err) = failure {
            return Err(err);
        }
        close?;
        Ok(reports)
    }

    /// Process a finite nonempty explicit sequence of weighted presentations.
    pub fn sequence_weighted(
        &mut self,
        presentations: &[Vec<InputAtom>],
        detailed_report: bool,
    ) -> Result<Vec<StepReport>> {
        if presentations.is_empty() {
            return Err(Error::Invalid("sequence must be nonempty".into()));
        }
        self.begin_sequence(false)?;
        let mut reports = Vec::new();
        reports
            .try_reserve(presentations.len())
            .map_err(|_| Error::Inexecutable("cannot reserve sequence reports".into()))?;
        let mut failure = None;
        for presentation in presentations {
            match self.sequence_step_weighted(presentation, detailed_report) {
                Ok(report) => reports.push(report),
                Err(err) => {
                    failure = Some(err);
                    break;
                }
            }
        }
        let close = self.end_sequence();
        if let Some(err) = failure {
            return Err(err);
        }
        close?;
        Ok(reports)
    }

    /// Canonical direct NETWORK -> NETWORK composition: every upstream
    /// depth presentation is consumed as an independent atomic sequence.
    pub fn consume(
        &mut self,
        present_family: &[OutputPresentation],
        detailed_report: bool,
    ) -> Result<Vec<StepReport>> {
        if self.sequence_open {
            return Err(Error::Invalid(
                "consume cannot be used while a sequence is open".into(),
            ));
        }
        self.invalidate_previous();
        let mut reports = Vec::new();
        reports
            .try_reserve(present_family.len())
            .map_err(|_| Error::Inexecutable("cannot reserve composition reports".into()))?;
        for presentation in present_family {
            let input: Vec<InputAtom> = presentation
                .iter()
                .map(|atom| InputAtom {
                    weight: atom.weight,
                    center: atom.center.clone(),
                    variance: atom.variance,
                })
                .collect();
            reports.push(self.step_weighted(&input, detailed_report)?);
        }
        self.invalidate_previous();
        Ok(reports)
    }

    fn step_atoms(&mut self, presentation: Vec<Atom>, detailed_report: bool) -> Result<StepReport> {
        let mut transformations = Vec::new();
        self.force_solvency(&mut transformations)?;
        let maintenance_open = self.maintenance_units()?;
        let layer_count_start = self.layers.len();
        let mut present_family: Vec<OutputPresentation> = Vec::new();
        let mut future_family: Vec<OutputPresentation> = Vec::new();
        let mut all_seed_requests: Vec<(Space, usize, Kernel64)> = Vec::new();
        let mut layer_reports = Vec::new();
        let mut temporal_reports = Vec::new();
        let mut contexts: Vec<Option<Kernel64>> = vec![None; layer_count_start];
        let mut frontier_requested = false;

        // Complete geometric recursion first. Predictive-private temporal
        // cognition observes the resulting contexts but never feeds back into
        // geometry in the same presentation.
        let mut current = presentation;
        for (layer_index, context_slot) in contexts.iter_mut().enumerate().take(layer_count_start) {
            if current.is_empty() {
                break;
            }
            let result = self.process_layer(layer_index, &current, detailed_report)?;
            *context_slot = result.context;
            if !result.knowledge.is_empty() {
                present_family.push(complete_output_presentation(
                    result.knowledge,
                    self.dimension,
                )?);
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

        if self.mode.predictive() {
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
                // before this presentation's private temporal learning phase.
                if let Some(context) = context {
                    let temporal_cells = &self.layers[layer_index].temporal_cells;
                    if !temporal_cells.is_empty() {
                        let d0 = norm2(&context.center);
                        let candidate_range =
                            first_coordinate_candidate_range(temporal_cells, context.center[0], d0);
                        let current_nonzero = context.center.iter().any(|&x| x != 0.0);
                        let mut matches = candidate_range.clone().filter_map(|index| {
                            let cell = &temporal_cells[index];
                            debug_assert_eq!(cell.center.len(), self.dimension * 2);
                            point_relative_gain_scalar(
                                &context.center,
                                d0,
                                current_nonzero,
                                &cell.center[..self.dimension],
                            )
                            .map(|gamma| (index, gamma))
                        });
                        if let Some(first) = matches.next() {
                            if let Some(second) = matches.next() {
                                let mut future_matches = Vec::new();
                                future_matches
                                    .try_reserve(candidate_range.len().min(temporal_cells.len()))
                                    .map_err(|_| {
                                        Error::Inexecutable(
                                            "cannot reserve predictive candidate matches".into(),
                                        )
                                    })?;
                                future_matches.push(first);
                                future_matches.push(second);
                                future_matches.extend(matches);
                                future_matches.sort_unstable_by(|&(a, _), &(b, _)| {
                                    cmp_scalar_vec(
                                        &temporal_cells[a].center[self.dimension..],
                                        &temporal_cells[b].center[self.dimension..],
                                    )
                                });

                                let mut group_start = 0;
                                while group_start < future_matches.len() {
                                    let (index, mut max_gamma) = future_matches[group_start];
                                    let target_slice =
                                        &temporal_cells[index].center[self.dimension..];
                                    let mut group_end = group_start + 1;
                                    while group_end < future_matches.len() {
                                        let (other_index, gamma) = future_matches[group_end];
                                        if temporal_cells[other_index].center[self.dimension..]
                                            != *target_slice
                                        {
                                            break;
                                        }
                                        max_gamma = max_gamma.max(gamma);
                                        group_end += 1;
                                    }
                                    let target = scalar_vec_to_f64(target_slice);
                                    future_family.push(complete_output_presentation(
                                        vec![OutputAtom {
                                            weight: context.weight * max_gamma,
                                            center: target,
                                            variance: 0.0,
                                        }],
                                        self.dimension,
                                    )?);
                                    group_start = group_end;
                                }
                            } else {
                                let (index, gamma) = first;
                                let target = scalar_vec_to_f64(
                                    &temporal_cells[index].center[self.dimension..],
                                );
                                future_family.push(complete_output_presentation(
                                    vec![OutputAtom {
                                        weight: context.weight * gamma,
                                        center: target,
                                        variance: 0.0,
                                    }],
                                    self.dimension,
                                )?);
                            }
                        }
                    }
                }

                if let (Some(previous), Some(context)) = (previous.as_ref(), context) {
                    let atom = self.temporal_atom(previous, context)?;
                    let result = self.process_temporal(layer_index, &[atom], detailed_report)?;
                    transformations.extend(result.transformations);
                    all_seed_requests.extend(
                        result
                            .seed_requests
                            .into_iter()
                            .map(|seed| (Space::Temporal, layer_index, seed)),
                    );
                    if let Some(report) = result.report {
                        temporal_reports.push(report);
                    }
                }

                // P_k is causal state, not learned memory: it advances even at
                // eta=0, but only inside the explicit sequence boundary.
                let next_previous = context_slot
                    .as_ref()
                    .map(|context| self.project_previous(context))
                    .transpose()?;
                self.layers[layer_index].previous = next_previous;
            }
        }

        // One material transaction spans geometric and predictive-private
        // temporal seeds plus the optional frontier layer.
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
            let projected_center = scalar_vec_to_f64(&projected.center);
            if cells.iter().any(|cell| {
                concern_raw_kernel(
                    cell,
                    &projected_center,
                    projected.variance.to_f64(),
                    projected.norm2,
                )
                .0
            }) {
                continue;
            }
            match space {
                Space::Geometry => projected_geometry[layer_index].push(projected),
                Space::Temporal => projected_temporal[layer_index].push(projected),
            }
        }

        let mut geometric_seeds = 0usize;
        let mut temporal_seeds = 0usize;
        let mut net_new_geometry = 0usize;
        let mut net_new_temporal = 0usize;

        // Coalesce only the new requests first.  The existing Sigma state is
        // left untouched until the global material transaction is known to be
        // payable; rejection therefore allocates no full future copy.
        for layer_index in 0..layer_count_start {
            geometric_seeds = geometric_seeds
                .checked_add(projected_geometry[layer_index].len())
                .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
            temporal_seeds = temporal_seeds
                .checked_add(projected_temporal[layer_index].len())
                .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;

            if !projected_geometry[layer_index].is_empty() {
                let additions = mem::take(&mut projected_geometry[layer_index]);
                projected_geometry[layer_index] = coalesce_projected(additions)?;
                net_new_geometry = net_new_geometry
                    .checked_add(count_new_geometries(
                        &self.layers[layer_index].sigma,
                        &projected_geometry[layer_index],
                    ))
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
            }
            if !projected_temporal[layer_index].is_empty() {
                let additions = mem::take(&mut projected_temporal[layer_index]);
                projected_temporal[layer_index] = coalesce_projected(additions)?;
                net_new_temporal = net_new_temporal
                    .checked_add(count_new_geometries(
                        &self.layers[layer_index].temporal_sigma,
                        &projected_temporal[layer_index],
                    ))
                    .ok_or_else(|| Error::Invalid("seed accounting overflow".into()))?;
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
            // Promotions and other transformations earlier in this same step
            // may have changed maintenance.  Solvency is decided against the
            // state that will actually receive the transaction, not the
            // administrative snapshot taken at step entry.
            let maintenance_before_growth = self.maintenance_units()?;
            let payable = maintenance_before_growth
                .checked_add(growth_cost)
                .is_some_and(|units| units <= self.budget_units);
            if payable {
                for layer_index in 0..layer_count_start {
                    if !projected_geometry[layer_index].is_empty() {
                        let existing = mem::take(&mut self.layers[layer_index].sigma);
                        let additions = mem::take(&mut projected_geometry[layer_index]);
                        let clock = self.layers[layer_index].cell_decay.clone();
                        let epoch = clock.epoch();
                        self.layers[layer_index].sigma =
                            merge_projected(existing, additions, clock, epoch)?;
                    }
                    if !projected_temporal[layer_index].is_empty() {
                        let existing = mem::take(&mut self.layers[layer_index].temporal_sigma);
                        let additions = mem::take(&mut projected_temporal[layer_index]);
                        let clock = self.layers[layer_index].temporal_decay.clone();
                        let epoch = clock.epoch();
                        self.layers[layer_index].temporal_sigma =
                            merge_projected(existing, additions, clock, epoch)?;
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

        let step_index = self.steps_seen;
        self.steps_seen = self.steps_seen.saturating_add(1);
        let maintenance_end = self.maintenance_units()?;
        if maintenance_end > self.budget_units {
            return Err(Error::Inexecutable(
                "internal error: post-step state exceeds budget".into(),
            ));
        }

        future_family.sort_by(output_presentation_cmp);
        future_family.dedup();
        let readout = match self.mode {
            Mode::Geometry => Readout::Geometry {
                present: present_family,
            },
            Mode::Predictive => Readout::Predictive {
                present: present_family,
                future: future_family,
            },
        };
        Ok(StepReport {
            step_index,
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
        let mut atoms = Vec::new();
        atoms
            .try_reserve(value.len())
            .map_err(|_| Error::Inexecutable("cannot reserve presentation atoms".into()))?;
        for (i, x) in value.iter().enumerate() {
            if x.len() != self.dimension {
                return Err(Error::Invalid(format!(
                    "presentation[{i}] must have dimension {}",
                    self.dimension
                )));
            }
            let mut canonical = Vec::new();
            canonical
                .try_reserve(self.dimension)
                .map_err(|_| Error::Inexecutable("cannot reserve presentation vector".into()))?;
            let mut norm_sum = 0.0;
            let mut norm_correction = 0.0;
            let mut zero = true;
            for &component in x {
                if !component.is_finite() {
                    return Err(Error::Invalid(format!(
                        "presentation[{i}] must contain only finite reals"
                    )));
                }
                if self.beta > 0.0 && component.abs() > S::max_finite().to_f64() {
                    return Err(Error::Invalid(format!(
                        "presentation[{i}] exceeds persistent {} range while learning is enabled",
                        S::NAME
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

    fn weighted_presentation(&self, value: &[InputAtom]) -> Result<Vec<Atom>> {
        if value.is_empty() {
            return Err(Error::Invalid(
                "external presentation must be a nonempty sequence".into(),
            ));
        }
        let mut atoms = Vec::new();
        atoms.try_reserve(value.len()).map_err(|_| {
            Error::Inexecutable("cannot reserve weighted presentation atoms".into())
        })?;
        for (i, item) in value.iter().enumerate() {
            if !item.weight.is_finite() || item.weight <= 0.0 {
                return Err(Error::Invalid(format!(
                    "presentation[{i}].weight must be finite and positive"
                )));
            }
            if item.center.len() != self.dimension {
                return Err(Error::Invalid(format!(
                    "presentation[{i}].center must have dimension {}",
                    self.dimension
                )));
            }
            if !item.variance.is_finite() || item.variance < 0.0 {
                return Err(Error::Invalid(format!(
                    "presentation[{i}].variance must be finite and nonnegative"
                )));
            }
            if self.beta > 0.0 && item.variance > S::max_finite().to_f64() {
                return Err(Error::Invalid(format!(
                    "presentation[{i}].variance exceeds persistent {} range while learning is enabled",
                    S::NAME
                )));
            }
            let mut center = Vec::new();
            center.try_reserve(self.dimension).map_err(|_| {
                Error::Inexecutable("cannot reserve weighted presentation vector".into())
            })?;
            let mut norm_sum = 0.0;
            let mut norm_correction = 0.0;
            let mut zero = true;
            for &component in &item.center {
                if !component.is_finite() {
                    return Err(Error::Invalid(format!(
                        "presentation[{i}].center must contain only finite reals"
                    )));
                }
                if self.beta > 0.0 && component.abs() > S::max_finite().to_f64() {
                    return Err(Error::Invalid(format!(
                        "presentation[{i}].center exceeds persistent {} range while learning is enabled",
                        S::NAME
                    )));
                }
                let component = structural_zero(component);
                zero &= component == 0.0;
                compensated_add(&mut norm_sum, &mut norm_correction, component * component);
                center.push(component);
            }
            atoms.push(Atom {
                x: Arc::from(center),
                r: structural_zero(item.weight),
                variance: structural_zero(item.variance),
                norm2: compensated_finish(norm_sum, norm_correction),
                zero,
            });
        }
        let total = stable_sum(atoms.iter().map(|atom| atom.r));
        if !(total > 0.0 && total <= 1.0) {
            return Err(Error::Invalid(
                "presentation mass must lie in (0, 1]".into(),
            ));
        }
        Ok(coalesce_atoms(atoms))
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
        if cell_epoch == u64::MAX {
            let lambda = layer.cell_decay.lambda();
            for kernel in &mut layer.sigma {
                kernel.materialize_weight_at(cell_epoch, lambda);
                kernel.decay_epoch = 0;
            }
            for cell in &mut layer.cells {
                cell.materialize_weight_at(cell_epoch, lambda);
                cell.decay_epoch = 0;
            }
            layer.cell_decay.epoch.store(0, AtomicOrdering::Relaxed);
            cell_epoch = 0;
        }
        let cell_decay_clock = layer.cell_decay.clone();
        let cell_count_before = layer.cells.len();
        let sigma_count_before = layer.sigma.len();

        // External singletons keep the specialised V=0 path. Internal
        // contextual atoms have V>0 and use the general total-variance path.
        let point_single = presentation.len() == 1 && presentation[0].variance == 0.0;
        // All fallible scratch reservation happens before persistent vectors
        // are moved out of the layer.  Allocation failure therefore leaves the
        // cognitive state untouched.
        if self.beta > 0.0 {
            // Reserve for the largest population before moving either CELL or
            // Sigma out of persistent state. Later scratch resets can then be
            // allocation-free for this compartment, so allocation failure
            // cannot strand cognition in a transient local vector.
            targets.reset(
                cell_count_before.max(sigma_count_before),
                dimension,
                !point_single,
            )?;
        }
        let mut old_cells = mem::take(&mut layer.cells);
        let old_sigma = mem::take(&mut layer.sigma);
        debug_assert!(old_cells.iter().all(|kernel| !kernel.dirty));
        debug_assert!(old_sigma.iter().all(|kernel| !kernel.dirty));
        let mut cell_received = detailed.then(|| vec![0.0; old_cells.len()]);
        let mut context_contributions: Vec<(usize, f64)> = Vec::new();
        let mut recognised_atoms = 0usize;
        let mut recognition_count = 0usize;

        // CONCERN -> RECOGNISED KNOWLEDGE -> ALLOCATE from the frozen CELL
        // state. Present knowledge is partitioned only by geometric CONCERN
        // gain. Historical CELL support enters ALLOCATE only and therefore has
        // no authority over the vertical context or external present readout.
        for (atom_index, atom) in presentation.iter().enumerate() {
            if atom.zero {
                unknown.push(atom_index);
                continue;
            }
            let d0 = atom.norm2;
            concerned.clear();
            let candidate_range = first_coordinate_candidate_range(&old_cells, atom.x[0], d0);
            for ci in candidate_range {
                let (ok, gain, distance2) =
                    concern_scalar(&old_cells[ci], &atom.x, atom.variance, d0);
                if ok {
                    concerned.push((ci, gain));
                    if point_single && self.beta > 0.0 {
                        targets.mark_single(ci, distance2);
                    }
                }
            }
            if concerned.is_empty() {
                unknown.push(atom_index);
                continue;
            }

            recognised_atoms += 1;

            // Quotient exact recognised CELL values by center without building
            // an intermediate list.  Equal centers are contiguous in canonical
            // CELL order and have the same CONCERN gain.
            if space == Space::Geometry {
                let unique_count = distinct_concerned_centers(&old_cells, &concerned);
                recognition_count += unique_count;
                if unique_count == 1 {
                    context_contributions.push((concerned[0].0, atom.r));
                } else {
                    append_gain_weighted_knowledge(
                        &old_cells,
                        &concerned,
                        atom.r,
                        &mut context_contributions,
                    );
                }
            }

            // Learning authority remains support-weighted ALLOCATE, but it has
            // no observable role at eta=0 unless a detailed responsibility
            // report was explicitly requested.  A single winner has rho=r
            // exactly, independent of support history.
            if self.beta > 0.0 || detailed {
                if concerned.len() == 1 {
                    let ci = concerned[0].0;
                    let rho = atom.r;
                    if self.beta > 0.0 {
                        old_cells[ci].materialize_weight_at(cell_epoch, self.lambda);
                        if point_single {
                            targets.set_single_weight(ci, rho);
                        } else {
                            targets.add_atom(ci, atom_index, rho);
                        }
                    }
                    if let Some(received) = &mut cell_received {
                        received[ci] += rho;
                    }
                } else {
                    normalize_allocate_scores(
                        &mut concerned,
                        &mut old_cells,
                        cell_epoch,
                        self.lambda,
                    );
                    let denominator = stable_sum(concerned.iter().map(|&(_, score)| score));
                    if !(denominator > 0.0 && denominator.is_finite()) {
                        return Err(Error::Inexecutable(
                            "internal error: ALLOCATE normalization failed".into(),
                        ));
                    }
                    for &(ci, score) in &concerned {
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
                    }
                }
            }
        }

        if space == Space::Geometry {
            coalesce_knowledge_contributions(&old_cells, &mut context_contributions);
        }
        let context = if space == Space::Geometry {
            build_context_kernel(&old_cells, &context_contributions, dimension)
        } else {
            None
        };
        let knowledge = if space == Space::Geometry {
            build_output_knowledge(&old_cells, &context_contributions)
        } else {
            Vec::new()
        };

        // Context is frozen from L^- before local learning. A singleton
        // context (V=0) has no vertical authority; neither does an exactly
        // zero-centered context, because Auxein has no canonical direction
        // for such a symmetric relation.
        let context_emitted = context
            .as_ref()
            .is_some_and(|kernel| kernel.variance > 0.0 && !zero_f64_vec(&kernel.center));
        let output = context
            .as_ref()
            .filter(|kernel| kernel.variance > 0.0 && !zero_f64_vec(&kernel.center))
            .map(|kernel| {
                vec![Atom {
                    x: Arc::from(kernel.center.clone()),
                    r: kernel.weight,
                    variance: kernel.variance,
                    norm2: norm2(&kernel.center),
                    zero: false,
                }]
            })
            .unwrap_or_default();

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
                    output_mass: context.as_ref().map_or(0.0, |k| k.weight),
                    context_center: context.as_ref().map(|k| k.center.clone()),
                    context_variance: context.as_ref().map(|k| k.variance),
                    recognition_count,
                    knowledge_mass: stable_sum(knowledge.iter().map(|atom| atom.weight)),
                    present_atom_count: if knowledge.is_empty() {
                        0
                    } else {
                        let mass = stable_sum(knowledge.iter().map(|atom| atom.weight));
                        knowledge.len() + usize::from(mass < 1.0)
                    },
                    present_mass: if knowledge.is_empty() { 0.0 } else { 1.0 },
                    cell_responsibility_mass: cell_received.unwrap_or_default(),
                })
            } else {
                None
            };
            return Ok(LayerResult {
                output,
                context,
                knowledge,
                seed_requests: Vec::new(),
                transformations: Vec::new(),
                report,
            });
        }

        let mut current_cells = old_cells;
        let next_cell_epoch = cell_epoch + 1;
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

        let mut seed_requests = Vec::new();
        let mut updated_sigma = old_sigma;

        if unknown.is_empty() {
            // Drop CELL-phase target indices before reusing the scratch for
            // Sigma. No private kernel can be touched without an unknown atom.
            targets.reset(0, dimension, false)?;
        } else {
            targets.reset(updated_sigma.len(), dimension, !point_single)?;
            for &atom_index in &unknown {
                let atom = &presentation[atom_index];
                if atom.zero {
                    continue;
                }
                let d0 = atom.norm2;
                concerned.clear();
                let candidate_range =
                    first_coordinate_candidate_range(&updated_sigma, atom.x[0], d0);
                for si in candidate_range {
                    let sigma = &updated_sigma[si];
                    let (ok, gain, distance2) = concern_scalar(sigma, &atom.x, atom.variance, d0);
                    if ok {
                        concerned.push((si, gain));
                        if point_single {
                            targets.mark_single(si, distance2);
                        }
                    }
                }
                if concerned.is_empty() {
                    seed_requests.push(Kernel64 {
                        weight: self.beta * atom.r,
                        center: atom.x.to_vec(),
                        variance: atom.variance,
                    });
                    continue;
                }

                if concerned.len() == 1 {
                    let si = concerned[0].0;
                    updated_sigma[si].materialize_weight_at(cell_epoch, self.lambda);
                    if point_single {
                        targets.set_single_weight(si, atom.r);
                    } else {
                        targets.add_atom(si, atom_index, atom.r);
                    }
                } else {
                    normalize_allocate_scores(
                        &mut concerned,
                        &mut updated_sigma,
                        cell_epoch,
                        self.lambda,
                    );
                    let denominator = stable_sum(concerned.iter().map(|&(_, score)| score));
                    if !(denominator > 0.0 && denominator.is_finite()) {
                        return Err(Error::Inexecutable(
                            "internal error: Sigma ALLOCATE normalization failed".into(),
                        ));
                    }
                    for &(si, score) in &concerned {
                        let tau = atom.r * score / denominator;
                        if point_single {
                            targets.set_single_weight(si, tau);
                        } else {
                            targets.add_atom(si, atom_index, tau);
                        }
                    }
                }
            }

            // Sigma shares the compartment clock with CELLs. Untouched private
            // kernels keep their old epoch and are never multiplied merely for
            // existing; only actual recipients are materialised and advanced.
            if point_single {
                targets.apply_single_cell_population(
                    &mut updated_sigma,
                    &presentation[0].x,
                    self.beta,
                    self.lambda,
                    next_cell_epoch,
                )?;
            } else {
                targets.apply_cell_population(
                    &mut updated_sigma,
                    presentation,
                    self.beta,
                    self.lambda,
                    next_cell_epoch,
                )?;
            }
        }

        let sigma_indices_stable = retain_touched_nonzero(&mut updated_sigma, &targets.touched);
        let (mut updated_sigma, sigma_order_stable) =
            coalesce_touched(updated_sigma, &targets.touched, sigma_indices_stable)?;

        let mut transformations = Vec::new();
        let mut promoted = Vec::new();
        if !targets.touched.is_empty() {
            if sigma_order_stable {
                let has_promotion = targets.touched.iter().copied().any(|index| {
                    index < updated_sigma.len()
                        && updated_sigma[index].weight.to_f64() > self.beta
                        && !zero_scalar_vec(&updated_sigma[index].center)
                });
                if has_promotion {
                    let mut remaining = Vec::with_capacity(updated_sigma.len());
                    let mut touched_cursor = 0usize;
                    for (index, kernel) in updated_sigma.into_iter().enumerate() {
                        while touched_cursor < targets.touched.len()
                            && targets.touched[touched_cursor] < index
                        {
                            touched_cursor += 1;
                        }
                        let touched = touched_cursor < targets.touched.len()
                            && targets.touched[touched_cursor] == index;
                        if touched
                            && kernel.weight.to_f64() > self.beta
                            && !zero_scalar_vec(&kernel.center)
                        {
                            promoted.push(kernel);
                        } else {
                            remaining.push(kernel);
                        }
                    }
                    updated_sigma = remaining;
                }
            } else {
                // Reordering destroys the old index map. This is a rare path;
                // scan logical current weights rather than guessing provenance.
                let mut remaining = Vec::with_capacity(updated_sigma.len());
                for kernel in updated_sigma {
                    if kernel.weight() > self.beta && !zero_scalar_vec(&kernel.center) {
                        promoted.push(kernel);
                    } else {
                        remaining.push(kernel);
                    }
                }
                updated_sigma = remaining;
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
        if updated_sigma.is_empty() && seed_requests.is_empty() {
            layer.sigma.clear();
        } else {
            let changed_indices = &targets.changed;
            let global_recheck =
                !changed_indices.is_empty() || promoted_count > 0 || !sigma_order_stable;

            if global_recheck {
                updated_sigma.retain_mut(|sigma| {
                    let candidate_range = first_coordinate_candidate_range(
                        &current_cells,
                        sigma.center[0].to_f64(),
                        sigma.norm2,
                    );
                    let sigma_center = scalar_vec_to_f64(&sigma.center);
                    let sigma_variance = sigma.variance.to_f64();
                    let sigma_norm2 = sigma.norm2;
                    let covered = if sigma.dirty {
                        current_cells[candidate_range].iter().any(|cell| {
                            concern_raw_kernel(cell, &sigma_center, sigma_variance, sigma_norm2).0
                        })
                    } else {
                        let first =
                            changed_indices.partition_point(|&ci| ci < candidate_range.start);
                        let last = changed_indices.partition_point(|&ci| ci < candidate_range.end);
                        changed_indices[first..last].iter().any(|&ci| {
                            concern_raw_kernel(
                                &current_cells[ci],
                                &sigma_center,
                                sigma_variance,
                                sigma_norm2,
                            )
                            .0
                        })
                    };
                    if !covered {
                        sigma.dirty = false;
                    }
                    !covered
                });
            } else if !targets.touched.is_empty() {
                // No CELL geometry changed. Only Sigma kernels whose own
                // geometry moved can have become covered; untouched kernels
                // cannot change status. Fast path is independent of |Sigma|.
                let mut covered = Vec::new();
                for &index in &targets.touched {
                    if index >= updated_sigma.len() || !updated_sigma[index].dirty {
                        continue;
                    }
                    let sigma = &updated_sigma[index];
                    let candidate_range = first_coordinate_candidate_range(
                        &current_cells,
                        sigma.center[0].to_f64(),
                        sigma.norm2,
                    );
                    let sigma_center = scalar_vec_to_f64(&sigma.center);
                    if current_cells[candidate_range].iter().any(|cell| {
                        concern_raw_kernel(
                            cell,
                            &sigma_center,
                            sigma.variance.to_f64(),
                            sigma.norm2,
                        )
                        .0
                    }) {
                        covered.push(index);
                    }
                }
                if covered.is_empty() {
                    for &index in &targets.touched {
                        if index < updated_sigma.len() {
                            updated_sigma[index].dirty = false;
                        }
                    }
                } else {
                    let mut cursor = 0usize;
                    let mut index = 0usize;
                    updated_sigma.retain(|_| {
                        let remove = cursor < covered.len() && covered[cursor] == index;
                        if remove {
                            cursor += 1;
                        }
                        index += 1;
                        !remove
                    });
                    for kernel in &mut updated_sigma {
                        kernel.dirty = false;
                    }
                }
            }
            layer.sigma = updated_sigma;

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
                output_mass: context.as_ref().map_or(0.0, |k| k.weight),
                context_center: context.as_ref().map(|k| k.center.clone()),
                context_variance: context.as_ref().map(|k| k.variance),
                recognition_count,
                knowledge_mass: stable_sum(knowledge.iter().map(|atom| atom.weight)),
                present_atom_count: if knowledge.is_empty() {
                    0
                } else {
                    let mass = stable_sum(knowledge.iter().map(|atom| atom.weight));
                    knowledge.len() + usize::from(mass < 1.0)
                },
                present_mass: if knowledge.is_empty() { 0.0 } else { 1.0 },
                cell_responsibility_mass: cell_received.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(LayerResult {
            output,
            context,
            knowledge,
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

    fn temporal_atom(&self, previous: &Kernel64, current: &Kernel64) -> Result<Atom> {
        let capacity = self
            .dimension
            .checked_mul(2)
            .ok_or_else(|| Error::Inexecutable("temporal dimension overflow".into()))?;
        let mut center = Vec::new();
        center
            .try_reserve(capacity)
            .map_err(|_| Error::Inexecutable("cannot reserve temporal atom".into()))?;
        center.extend_from_slice(&previous.center);
        center.extend_from_slice(&current.center);
        let norm2 = norm2(&center);
        let zero = zero_f64_vec(&center);
        Ok(Atom {
            x: Arc::from(center),
            r: previous.weight * current.weight,
            variance: previous.variance + current.variance,
            norm2,
            zero,
        })
    }

    fn project_previous(&self, context: &Kernel64) -> Result<Kernel<S>> {
        project_kernel::<S>(context.clone())
    }

    fn invalidate_previous(&mut self) {
        if self.mode.predictive() {
            for layer in &mut self.layers {
                layer.previous = None;
            }
        }
    }

    fn layer_has_cells(&self, layer: &Layer<S>) -> bool {
        !layer.cells.is_empty() || (self.mode.predictive() && !layer.temporal_cells.is_empty())
    }

    fn force_solvency(&mut self, transformations: &mut Vec<Transformation>) -> Result<()> {
        let mut simulated = self.maintenance_units()?;
        if simulated <= self.budget_units {
            return Ok(());
        }

        // Work in progress is discarded simultaneously in both spaces.  Drop
        // the allocations as well: forced contraction is a memory-pressure
        // event, so retaining the high-water capacity would defeat its purpose
        // in a long-running process.
        let removed_sigma = self.layers.iter().fold(0usize, |total, layer| {
            total
                .saturating_add(layer.sigma.len())
                .saturating_add(if self.mode.predictive() {
                    layer.temporal_sigma.len()
                } else {
                    0
                })
        });
        if removed_sigma > 0 {
            for layer in &mut self.layers {
                layer.sigma = Vec::new();
                if self.mode.predictive() {
                    layer.temporal_sigma = Vec::new();
                }
            }
            transformations.push(Transformation::ClearSigma {
                count: removed_sigma,
            });
        }

        let mut trimmed = 0usize;
        while self.layers.len() > 1
            && self
                .layers
                .last()
                .is_some_and(|layer| !self.layer_has_cells(layer))
        {
            self.layers.pop();
            trimmed = trimmed.saturating_add(1);
        }
        if trimmed > 0 {
            transformations.push(Transformation::TrimLayers { count: trimmed });
        }
        simulated = self.maintenance_units()?;
        if simulated <= self.budget_units {
            self.invalidate_previous();
            return Ok(());
        }

        // One K ordering spans geometric and temporal knowledge. Equal K is
        // destroyed as a whole wave independently of its compartment.  Once
        // K is sorted, material simulation is purely decremental: do not
        // rescan every layer after every wave.
        let mut valued: Vec<(f64, usize, Space)> = Vec::new();
        let mut geometric_counts: Vec<usize> = self.layers.iter().map(|l| l.cells.len()).collect();
        let mut temporal_counts: Vec<usize> = self
            .layers
            .iter()
            .map(|l| {
                if self.mode.predictive() {
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
            if self.mode.predictive() {
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
            if simulated > self.budget_units {
                return Err(Error::Inexecutable(
                    "minimal Auxein state exceeds the execution budget".into(),
                ));
            }
            return Ok(());
        }
        valued.sort_by(|a, b| a.0.total_cmp(&b.0));

        let geometry_units = self.kernel_units()?;
        let temporal_units = self.temporal_kernel_units()?;
        let layer_units = self.layer_units()?;
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
                let units = match space {
                    Space::Geometry => {
                        geometric_counts[li] = geometric_counts[li].saturating_sub(1);
                        geometry_units
                    }
                    Space::Temporal => {
                        temporal_counts[li] = temporal_counts[li].saturating_sub(1);
                        temporal_units
                    }
                };
                simulated = simulated.checked_sub(units).ok_or_else(|| {
                    Error::Invalid("material accounting underflow during contraction".into())
                })?;
                removed_cells = removed_cells.saturating_add(1);
                stop += 1;
            }
            waves = waves.saturating_add(1);
            while active_layers > 1
                && geometric_counts[active_layers - 1] == 0
                && temporal_counts[active_layers - 1] == 0
            {
                active_layers -= 1;
                simulated = simulated.checked_sub(layer_units).ok_or_else(|| {
                    Error::Invalid("material accounting underflow during layer trim".into())
                })?;
            }
            cutoff = Some(k);
            if simulated <= self.budget_units {
                break;
            }
            position = stop;
        }
        let cutoff = cutoff.ok_or_else(|| {
            Error::Inexecutable("forced solvency found no removable knowledge".into())
        })?;

        for layer in &mut self.layers {
            layer.cells.retain(|cell| Self::cell_value(cell) > cutoff);
            layer.cells.shrink_to_fit();
            if self.mode.predictive() {
                layer
                    .temporal_cells
                    .retain(|cell| Self::cell_value(cell) > cutoff);
                layer.temporal_cells.shrink_to_fit();
            }
        }
        let mut trimmed_after = 0usize;
        while self.layers.len() > 1
            && self
                .layers
                .last()
                .is_some_and(|layer| !self.layer_has_cells(layer))
        {
            self.layers.pop();
            trimmed_after = trimmed_after.saturating_add(1);
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
                    if self.mode.predictive() {
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
                    if self.mode.predictive() {
                        l.temporal_sigma.len()
                    } else {
                        0
                    }
                })
                .collect(),
            previous_context_per_layer: self
                .layers
                .iter()
                .map(|l| self.mode.predictive() && l.previous.is_some())
                .collect(),
            maintenance_units: maintenance,
            budget: self.budget_equivalent()?,
            budget_units: self.budget_units,
            budget_margin_units: self.budget_units as i128 - maintenance as i128,
            is_solvent: maintenance <= self.budget_units,
        })
    }

    fn write_json_sink(&self, out: &mut impl JsonSink) {
        out.push('{');
        out.push_str("\"format_version\":5,");
        out.push_str("\"dimension\":");
        out.push_str(&self.dimension.to_string());
        out.push_str(",\"scalar\":");
        sink_quote(out, S::NAME);
        out.push_str(",\"memory\":");
        push_float(out, self.memory.to_f64());
        out.push_str(",\"eta\":");
        push_float(out, self.eta.to_f64());
        out.push_str(",\"mode\":");
        sink_quote(out, self.mode.as_str());
        out.push_str(",\"steps_seen\":");
        out.push_str(&self.steps_seen.to_string());
        out.push_str(",\"layers\":[");
        for (li, layer) in self.layers.iter().enumerate() {
            if li > 0 {
                out.push(',');
            }
            out.push_str("{\"sigma\":[");
            write_kernel_list(out, &layer.sigma);
            out.push_str("],\"cells\":[");
            write_kernel_list(out, &layer.cells);
            if self.mode.predictive() {
                out.push_str("],\"temporal_sigma\":[");
                write_kernel_list(out, &layer.temporal_sigma);
                out.push_str("],\"temporal_cells\":[");
                write_kernel_list(out, &layer.temporal_cells);
                out.push_str("],\"previous\":");
                if let Some(previous) = &layer.previous {
                    write_kernel(out, previous);
                } else {
                    out.push_str("null");
                }
                out.push('}');
            } else {
                out.push_str("]}");
            }
        }
        out.push_str("]}");
    }

    pub fn export_json(&self) -> String {
        let reserve = self
            .maintenance_units()
            .ok()
            .and_then(|units| usize::try_from(units).ok())
            .and_then(|units| units.checked_add(256))
            .unwrap_or(256);
        let mut out = String::with_capacity(reserve);
        self.write_json_sink(&mut out);
        out
    }

    /// Stream canonical format_version=5 JSON without constructing a second
    /// full in-memory copy of the persistent state.
    pub fn write_json<W: IoWrite>(&self, writer: &mut W) -> Result<()> {
        let mut sink = IoJsonSink::new(writer);
        self.write_json_sink(&mut sink);
        sink.finish()
    }

    pub fn from_json(json_text: &str, budget: Budget) -> Result<Self> {
        let parsed = json::parse(json_text)?;
        let state = ParsedState::from_json(parsed)?;
        if state.scalar != S::NAME {
            return Err(Error::Invalid(format!(
                "state scalar is {}, requested engine is {}",
                state.scalar,
                S::NAME
            )));
        }
        Self::from_parsed_state(state, budget)
    }

    fn from_parsed_state(state: ParsedState, budget: Budget) -> Result<Self> {
        let mut network =
            Self::new_with_mode(state.dimension, state.memory, state.eta, state.mode, budget)?;
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
            let mut sigma = load_kernel_list::<S>(
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
            for kernel in &mut sigma {
                kernel.bind_decay_clock(clock.clone(), 0);
            }
            let mut cells = cells;
            for cell in &mut cells {
                cell.bind_decay_clock(clock.clone(), 0);
            }

            let mut temporal_sigma = load_kernel_list::<S>(
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
            for kernel in &mut temporal_sigma {
                kernel.bind_decay_clock(temporal_clock.clone(), 0);
            }
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
            if self.mode.predictive() {
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
    ) -> Result<Self> {
        Self::new_with_mode(scalar, dimension, memory, eta, Mode::Geometry, budget)
    }

    pub fn new_with_mode(
        scalar: &str,
        dimension: usize,
        memory: f64,
        eta: f64,
        mode: Mode,
        budget: Budget,
    ) -> Result<Self> {
        match scalar {
            "f32" => Ok(Self::F32(Auxein::new_with_mode(
                dimension, memory, eta, mode, budget,
            )?)),
            "f64" => Ok(Self::F64(Auxein::new_with_mode(
                dimension, memory, eta, mode, budget,
            )?)),
            _ => Err(Error::Invalid("scalar must be 'f32' or 'f64'".into())),
        }
    }

    pub fn from_json(json_text: &str, budget: Budget) -> Result<Self> {
        let parsed = json::parse(json_text)?;
        let state = ParsedState::from_json(parsed)?;
        match state.scalar.as_str() {
            "f32" => Ok(Self::F32(Auxein::<f32>::from_parsed_state(state, budget)?)),
            "f64" => Ok(Self::F64(Auxein::<f64>::from_parsed_state(state, budget)?)),
            _ => Err(Error::Invalid("state.scalar must be 'f32' or 'f64'".into())),
        }
    }

    pub fn step(&mut self, presentation: &[Vec<f64>], detailed: bool) -> Result<StepReport> {
        match self {
            Self::F32(n) => n.step(presentation, detailed),
            Self::F64(n) => n.step(presentation, detailed),
        }
    }

    pub fn step_weighted(
        &mut self,
        presentation: &[InputAtom],
        detailed: bool,
    ) -> Result<StepReport> {
        match self {
            Self::F32(n) => n.step_weighted(presentation, detailed),
            Self::F64(n) => n.step_weighted(presentation, detailed),
        }
    }

    pub fn begin_sequence(&mut self, resume: bool) -> Result<()> {
        match self {
            Self::F32(n) => n.begin_sequence(resume),
            Self::F64(n) => n.begin_sequence(resume),
        }
    }

    pub fn end_sequence(&mut self) -> Result<()> {
        match self {
            Self::F32(n) => n.end_sequence(),
            Self::F64(n) => n.end_sequence(),
        }
    }

    pub fn sequence_step(
        &mut self,
        presentation: &[Vec<f64>],
        detailed: bool,
    ) -> Result<StepReport> {
        match self {
            Self::F32(n) => n.sequence_step(presentation, detailed),
            Self::F64(n) => n.sequence_step(presentation, detailed),
        }
    }

    pub fn sequence_step_weighted(
        &mut self,
        presentation: &[InputAtom],
        detailed: bool,
    ) -> Result<StepReport> {
        match self {
            Self::F32(n) => n.sequence_step_weighted(presentation, detailed),
            Self::F64(n) => n.sequence_step_weighted(presentation, detailed),
        }
    }

    pub fn consume(
        &mut self,
        present_family: &[OutputPresentation],
        detailed: bool,
    ) -> Result<Vec<StepReport>> {
        match self {
            Self::F32(n) => n.consume(present_family, detailed),
            Self::F64(n) => n.consume(present_family, detailed),
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

    pub fn transient_memory_capacity_bytes(&self) -> usize {
        match self {
            Self::F32(n) => n.transient_memory_capacity_bytes(),
            Self::F64(n) => n.transient_memory_capacity_bytes(),
        }
    }

    pub fn compact_transient_memory(&mut self) {
        match self {
            Self::F32(n) => n.compact_transient_memory(),
            Self::F64(n) => n.compact_transient_memory(),
        }
    }

    pub fn export_json(&self) -> String {
        match self {
            Self::F32(n) => n.export_json(),
            Self::F64(n) => n.export_json(),
        }
    }

    pub fn write_json<W: IoWrite>(&self, writer: &mut W) -> Result<()> {
        match self {
            Self::F32(n) => n.write_json(writer),
            Self::F64(n) => n.write_json(writer),
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
                Mode::Predictive => exact_keys(
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
                Mode::Predictive => {
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
    if kernels.len() <= 8 || !d0.is_finite() || d0 <= 0.0 {
        return 0..kernels.len();
    }

    // Kernels are in canonical lexicographic center order.  The necessary
    // first-coordinate condition is monotone on either side of x0, so both
    // bounds are binary searches using the exact same squared comparison as
    // the full CONCERN test.
    let split = kernels.partition_point(|kernel| kernel.center[0].to_f64() < x0);
    let start = kernels[..split].partition_point(|kernel| {
        let delta = x0 - kernel.center[0].to_f64();
        delta * delta >= d0
    });
    let right = kernels[split..].partition_point(|kernel| {
        let delta = kernel.center[0].to_f64() - x0;
        delta * delta < d0
    });
    start..split + right
}

fn point_relative_gain_scalar<S: Scalar>(
    current: &[f64],
    current2: f64,
    current_nonzero: bool,
    source: &[S],
) -> Option<f64> {
    debug_assert_eq!(current.len(), source.len());
    let source2 = norm2_scalar(source);
    let source_nonzero = source.iter().any(|&x| x.to_f64() != 0.0);
    let extreme = !current2.is_finite()
        || !source2.is_finite()
        || current2.is_subnormal()
        || source2.is_subnormal()
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
            return None;
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
        if !(distance_scaled < current_scaled && distance_scaled < source_scaled) {
            return None;
        }
        return Some((current_scaled - distance_scaled) / current_scaled);
    }

    let distance2 = stable_sum(current.iter().zip(source).map(|(&a, &b)| {
        let d = a - b.to_f64();
        d * d
    }));
    if !(distance2 < current2 && distance2 < source2) {
        return None;
    }
    Some((current2 - distance2) / current2)
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

    // A concerned center necessarily satisfies ||C|| < 2||x||.  Rejecting
    // impossible energetic centers before touching D coordinates protects the
    // degenerate case where many kernels share the same indexed coordinate.
    let norm_limit = 4.0 * d0_center;
    if norm_limit.is_finite() && kernel.norm2 >= norm_limit {
        return (false, 0.0, 0.0);
    }

    // D_i < D_0 contains the same incoming variance on both sides, so cancel
    // it before arithmetic.  This avoids a false `inf < inf` at huge finite
    // variances and is exactly the canonical inequality.
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
    if geometric.partial_cmp(&d0_center) != Some(Ordering::Less) {
        return (false, d0_center - geometric, geometric);
    }
    let ok = positive_sum_lt(
        geometric,
        input_variance,
        kernel.norm2,
        kernel.variance.to_f64(),
    );
    (ok, d0_center - geometric, geometric)
}

#[inline]
fn positive_sum_lt(a: f64, b: f64, c: f64, d: f64) -> bool {
    debug_assert!(a >= 0.0 && b >= 0.0 && c >= 0.0 && d >= 0.0);
    if c.is_infinite() || d.is_infinite() {
        return !(a.is_infinite() || b.is_infinite());
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let left = a + b;
    let right = c + d;
    if left.is_finite() && right.is_finite() {
        return left < right;
    }
    let scale = a.max(b).max(c).max(d);
    if scale == 0.0 {
        return false;
    }
    a / scale + b / scale < c / scale + d / scale
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

fn coalesce_output_presentation(mut atoms: Vec<OutputAtom>) -> Vec<OutputAtom> {
    atoms.sort_by(|a, b| {
        cmp_vec(&a.center, &b.center).then_with(|| a.variance.total_cmp(&b.variance))
    });
    if !atoms
        .windows(2)
        .any(|pair| pair[0].center == pair[1].center && pair[0].variance == pair[1].variance)
    {
        return atoms;
    }
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        let center = atoms[i].center.clone();
        let variance = atoms[i].variance;
        let mut j = i + 1;
        while j < atoms.len() && atoms[j].center == center && atoms[j].variance == variance {
            j += 1;
        }
        let weight = stable_sum(atoms[i..j].iter().map(|atom| atom.weight));
        if weight > 0.0 {
            out.push(OutputAtom {
                weight,
                center,
                variance,
            });
        }
        i = j;
    }
    out
}

fn complete_output_presentation(
    mut atoms: Vec<OutputAtom>,
    dimension: usize,
) -> Result<OutputPresentation> {
    let already_canonical = atoms
        .windows(2)
        .all(|pair| output_geometry_cmp(&pair[0], &pair[1]) == Ordering::Less);
    if !already_canonical {
        atoms = coalesce_output_presentation(atoms);
    }
    let total = stable_sum(atoms.iter().map(|atom| atom.weight));
    let mut remainder = 1.0 - total;
    if remainder < 0.0 {
        if remainder.abs() <= 8.0 * f64::EPSILON {
            remainder = 0.0;
        } else {
            return Err(Error::Inexecutable(format!(
                "internal error: output presentation mass exceeds one (total={total:.17e})"
            )));
        }
    }
    if remainder > 0.0 {
        let zero = OutputAtom {
            weight: remainder,
            center: vec![0.0; dimension],
            variance: 0.0,
        };
        match atoms.binary_search_by(|atom| output_geometry_cmp(atom, &zero)) {
            Ok(index) => atoms[index].weight += remainder,
            Err(index) => atoms.insert(index, zero),
        }
    }
    Ok(atoms)
}

fn distinct_concerned_centers<S: Scalar>(cells: &[Kernel<S>], concerned: &[(usize, f64)]) -> usize {
    let mut count = 0usize;
    let mut previous: Option<&[S]> = None;
    for &(index, _) in concerned {
        let center = cells[index].center.as_slice();
        if previous.is_none_or(|value| value != center) {
            count += 1;
            previous = Some(center);
        }
    }
    count
}

fn append_gain_weighted_knowledge<S: Scalar>(
    cells: &[Kernel<S>],
    concerned: &[(usize, f64)],
    atom_mass: f64,
    out: &mut Vec<(usize, f64)>,
) {
    let mut gain_scale = 0.0f64;
    let mut previous: Option<&[S]> = None;
    for &(index, gain) in concerned {
        let center = cells[index].center.as_slice();
        if previous.is_none_or(|value| value != center) {
            gain_scale = gain_scale.max(gain);
            previous = Some(center);
        }
    }
    debug_assert!(gain_scale > 0.0 && gain_scale.is_finite());

    let mut denominator = 0.0;
    let mut correction = 0.0;
    previous = None;
    for &(index, gain) in concerned {
        let center = cells[index].center.as_slice();
        if previous.is_none_or(|value| value != center) {
            compensated_add(&mut denominator, &mut correction, gain / gain_scale);
            previous = Some(center);
        }
    }
    denominator = compensated_finish(denominator, correction);

    let mut assigned_sum = 0.0;
    let mut assigned_correction = 0.0;
    let mut seen = 0usize;
    let unique_count = distinct_concerned_centers(cells, concerned);
    previous = None;
    for &(index, gain) in concerned {
        let center = cells[index].center.as_slice();
        if previous.is_some_and(|value| value == center) {
            continue;
        }
        seen += 1;
        let weight = if seen == unique_count {
            (atom_mass - compensated_finish(assigned_sum, assigned_correction)).max(0.0)
        } else {
            let weight = atom_mass * (gain / gain_scale) / denominator;
            compensated_add(&mut assigned_sum, &mut assigned_correction, weight);
            weight
        };
        if weight > 0.0 {
            out.push((index, weight));
        }
        previous = Some(center);
    }
}

fn normalize_allocate_scores<S: Scalar>(
    concerned: &mut [(usize, f64)],
    cells: &mut [Kernel<S>],
    epoch: u64,
    lambda: f64,
) {
    let mut max_gain = 0.0f64;
    let mut max_weight = 0.0f64;
    for &(index, gain) in concerned.iter() {
        max_gain = max_gain.max(gain);
        max_weight = max_weight.max(cells[index].materialize_weight_at(epoch, lambda));
    }
    for (index, score) in concerned.iter_mut() {
        let weight = cells[*index].weight.to_f64();
        *score = (*score / max_gain) * (weight / max_weight);
    }
    let denominator = stable_sum(concerned.iter().map(|&(_, score)| score));
    if denominator > 0.0 && denominator.is_finite() {
        return;
    }
    normalize_allocate_scores_extreme(concerned, cells);
}

#[cold]
#[inline(never)]
fn normalize_allocate_scores_extreme<S: Scalar>(
    concerned: &mut [(usize, f64)],
    cells: &[Kernel<S>],
) {
    let mut max_log = f64::NEG_INFINITY;
    for &(index, gain) in concerned.iter() {
        max_log = max_log.max(gain.ln() + cells[index].weight.to_f64().ln());
    }
    for (index, score) in concerned.iter_mut() {
        *score = (score.ln() + cells[*index].weight.to_f64().ln() - max_log).exp();
    }
}

#[inline]
fn saturating_positive_scalar<S: Scalar>(value: f64) -> S {
    let min = S::min_positive().to_f64();
    let max = S::max_finite().to_f64();
    let bounded = if value.is_nan() || value.is_infinite() || value > max {
        max
    } else if value <= 0.0 {
        min
    } else {
        value.max(min)
    };
    <S as sealed::Sealed>::from_finite(bounded)
}
fn coalesce_knowledge_contributions<S: Scalar>(
    cells: &[Kernel<S>],
    contributions: &mut Vec<(usize, f64)>,
) {
    if contributions.len() <= 1 {
        return;
    }
    contributions.sort_unstable_by_key(|&(index, _)| index);
    let mut write = 0usize;
    let mut read = 0usize;
    while read < contributions.len() {
        let first_index = contributions[read].0;
        let center = cells[first_index].center.as_slice();
        let mut end = read + 1;
        while end < contributions.len() && cells[contributions[end].0].center.as_slice() == center {
            end += 1;
        }
        let weight = stable_sum(contributions[read..end].iter().map(|&(_, weight)| weight));
        contributions[write] = (first_index, weight);
        write += 1;
        read = end;
    }
    contributions.truncate(write);
}

fn build_output_knowledge<S: Scalar>(
    cells: &[Kernel<S>],
    contributions: &[(usize, f64)],
) -> Vec<OutputAtom> {
    contributions
        .iter()
        .filter(|&&(_, weight)| weight > 0.0)
        .map(|&(index, weight)| OutputAtom {
            weight,
            center: scalar_vec_to_f64(&cells[index].center),
            variance: 0.0,
        })
        .collect()
}

fn output_geometry_cmp(a: &OutputAtom, b: &OutputAtom) -> Ordering {
    cmp_vec(&a.center, &b.center).then_with(|| a.variance.total_cmp(&b.variance))
}

fn output_atom_cmp(a: &OutputAtom, b: &OutputAtom) -> Ordering {
    cmp_vec(&a.center, &b.center)
        .then_with(|| a.variance.total_cmp(&b.variance))
        .then_with(|| a.weight.total_cmp(&b.weight))
}

fn output_presentation_cmp(a: &OutputPresentation, b: &OutputPresentation) -> Ordering {
    for (left, right) in a.iter().zip(b) {
        let order = output_atom_cmp(left, right);
        if order != Ordering::Equal {
            return order;
        }
    }
    a.len().cmp(&b.len())
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
        let distance2 = robust_distance2_scalar_f64(&cells[index].center, &center);
        variance_terms.push(w * distance2);
    }
    let raw_variance = orderless_sum(&variance_terms, &mut partials) / weight;
    let scalar_max = S::max_finite().to_f64();
    let variance = if raw_variance.is_nan() || raw_variance < 0.0 {
        // All terms are nonnegative, so NaN here can only be a representational
        // overflow/cancellation artifact.  Preserve a finite persistent state.
        scalar_max
    } else if !raw_variance.is_finite() || raw_variance > scalar_max {
        scalar_max
    } else if raw_variance < S::min_positive().to_f64() {
        // We already proved above that at least two exact centers differ.  The
        // mathematical variance is therefore strictly positive even when its
        // host value is nonzero but would project to zero in persistent S.
        S::min_positive().to_f64()
    } else {
        raw_variance
    };
    Some(Kernel64 {
        weight,
        center,
        variance: structural_zero(variance),
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
        let weight =
            saturating_positive_scalar::<S>(stable_sum(kernels[i..j].iter().map(Kernel::weight)));
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

fn count_new_geometries<S: Scalar>(existing: &[Kernel<S>], additions: &[Kernel<S>]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut new_count = 0usize;
    while j < additions.len() {
        while i < existing.len()
            && kernel_geometry_cmp(&existing[i], &additions[j]) == Ordering::Less
        {
            i += 1;
        }
        if i == existing.len()
            || kernel_geometry_cmp(&existing[i], &additions[j]) != Ordering::Equal
        {
            new_count += 1;
        }
        j += 1;
    }
    new_count
}

fn merge_projected<S: Scalar>(
    existing: Vec<Kernel<S>>,
    additions: Vec<Kernel<S>>,
    clock: Arc<DecayClock>,
    epoch: u64,
) -> Result<Vec<Kernel<S>>> {
    let capacity = existing.len().saturating_add(additions.len());
    let mut out = Vec::new();
    out.try_reserve(capacity)
        .map_err(|_| Error::Inexecutable("cannot reserve Sigma growth memory".into()))?;
    let mut left = existing.into_iter().peekable();
    let mut right = additions.into_iter().peekable();
    loop {
        match (left.peek(), right.peek()) {
            (Some(a), Some(b)) => match kernel_geometry_cmp(a, b) {
                Ordering::Less => out.push(left.next().ok_or_else(|| {
                    Error::Inexecutable("internal Sigma merge state is invalid".into())
                })?),
                Ordering::Greater => {
                    let mut kernel = right.next().ok_or_else(|| {
                        Error::Inexecutable("internal Sigma merge state is invalid".into())
                    })?;
                    kernel.bind_decay_clock(clock.clone(), epoch);
                    out.push(kernel);
                }
                Ordering::Equal => {
                    let mut a = left.next().ok_or_else(|| {
                        Error::Inexecutable("internal Sigma merge state is invalid".into())
                    })?;
                    let b = right.next().ok_or_else(|| {
                        Error::Inexecutable("internal Sigma merge state is invalid".into())
                    })?;
                    a.materialize_weight_at(epoch, clock.lambda());
                    let weight = stable_sum([a.weight.to_f64(), b.weight.to_f64()]);
                    a.weight = saturating_positive_scalar::<S>(weight);
                    a.decay_epoch = epoch;
                    a.bind_decay_clock(clock.clone(), epoch);
                    a.dirty = false;
                    out.push(a);
                }
            },
            (Some(_), None) => {
                out.extend(left);
                break;
            }
            (None, Some(_)) => {
                for mut kernel in right {
                    kernel.bind_decay_clock(clock.clone(), epoch);
                    out.push(kernel);
                }
                break;
            }
            (None, None) => break,
        }
    }
    Ok(out)
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
        let weight = saturating_positive_scalar::<S>(stable_sum(
            kernels[i..j].iter().map(|kernel| kernel.weight.to_f64()),
        ));
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
    // Neumaier compensation assumes finite arithmetic.  Positive overflow is
    // a legitimate result for several structural sums (notably squared norms)
    // and must remain +inf rather than being turned into NaN by `inf - inf`.
    // Signed infinities or NaN are propagated explicitly; behavioral callers
    // either reject them or switch to their homogeneous fallback.
    if !sum.is_finite() || !x.is_finite() {
        *sum += x;
        *correction = 0.0;
        return;
    }
    let next = *sum + x;
    if !next.is_finite() {
        *sum = next;
        *correction = 0.0;
        return;
    }
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
    squared_norm_iter(v.iter().copied())
}

fn norm2_scalar<S: Scalar>(v: &[S]) -> f64 {
    squared_norm_iter(v.iter().map(|&x| x.to_f64()))
}

#[inline]
fn squared_norm_iter<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let term = value * value;
        if term.is_infinite() {
            return f64::INFINITY;
        }
        compensated_add(&mut sum, &mut correction, term);
        if sum.is_infinite() {
            return f64::INFINITY;
        }
    }
    compensated_finish(sum, correction)
}

// Squared Euclidean distance for finite coordinates.  The normal path avoids
// extra passes.  If a square overflows, scale homogeneously; if the true
// positive value lies below binary64, MIN_POSITIVE preserves the exact
// topological fact "distance > 0" without introducing a behavioral epsilon.
fn robust_distance2_scalar_f64<S: Scalar>(a: &[S], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    let mut any_nonzero = false;
    let mut needs_scaled = false;
    for (&left, &right) in a.iter().zip(b) {
        let d = left.to_f64() - right;
        any_nonzero |= d != 0.0;
        let term = d * d;
        if !term.is_finite() {
            needs_scaled = true;
            break;
        }
        compensated_add(&mut sum, &mut correction, term);
    }
    if !needs_scaled {
        let out = compensated_finish(sum, correction);
        return if out == 0.0 && any_nonzero {
            f64::MIN_POSITIVE
        } else {
            out
        };
    }

    let mut scale = 0.0f64;
    for (&left, &right) in a.iter().zip(b) {
        scale = scale.max((left.to_f64() - right).abs());
    }
    if scale == 0.0 {
        return 0.0;
    }
    let scaled = stable_sum(a.iter().zip(b).map(|(&left, &right)| {
        let d = (left.to_f64() - right) / scale;
        d * d
    }));
    let scale2 = scale * scale;
    if scale2.is_infinite() {
        return f64::INFINITY;
    }
    let out = scaled * scale2;
    if out == 0.0 && any_nonzero {
        f64::MIN_POSITIVE
    } else {
        out
    }
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

trait JsonSink {
    fn push_str(&mut self, text: &str);
    fn push(&mut self, ch: char);
}

impl JsonSink for String {
    #[inline]
    fn push_str(&mut self, text: &str) {
        String::push_str(self, text);
    }

    #[inline]
    fn push(&mut self, ch: char) {
        String::push(self, ch);
    }
}

struct IoJsonSink<'a, W: IoWrite> {
    writer: &'a mut W,
    error: Option<std::io::Error>,
}

impl<'a, W: IoWrite> IoJsonSink<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            error: None,
        }
    }

    fn finish(self) -> Result<()> {
        match self.error {
            Some(error) => Err(Error::Io(error.to_string())),
            None => Ok(()),
        }
    }

    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) {
        if self.error.is_none() {
            if let Err(error) = self.writer.write_all(bytes) {
                self.error = Some(error);
            }
        }
    }
}

impl<W: IoWrite> JsonSink for IoJsonSink<'_, W> {
    #[inline]
    fn push_str(&mut self, text: &str) {
        self.write_bytes(text.as_bytes());
    }

    #[inline]
    fn push(&mut self, ch: char) {
        let mut encoded = [0u8; 4];
        self.write_bytes(ch.encode_utf8(&mut encoded).as_bytes());
    }
}

fn sink_quote(out: &mut impl JsonSink, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_float(out: &mut impl JsonSink, value: f64) {
    if value == 0.0 {
        out.push_str("0.0");
    } else {
        out.push_str(&value.to_string());
    }
}

fn write_kernel_list<S: Scalar>(out: &mut impl JsonSink, kernels: &[Kernel<S>]) {
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

fn write_kernel<S: Scalar>(out: &mut impl JsonSink, kernel: &Kernel<S>) {
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

pub fn parse_weighted_presentation_json(text: &str) -> Result<Vec<InputAtom>> {
    let value = json::parse(text)?;
    let json::Value::Array(items) = value else {
        return Err(Error::Invalid(
            "weighted presentation must be a JSON array".into(),
        ));
    };
    if items.is_empty() {
        return Err(Error::Invalid("presentation must be nonempty".into()));
    }
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let json::Value::Array(parts) = item else {
            return Err(Error::Invalid(
                "weighted presentation items must be [W,C,V]".into(),
            ));
        };
        if parts.len() != 3 {
            return Err(Error::Invalid(
                "weighted presentation items must be [W,C,V]".into(),
            ));
        }
        let mut it = parts.into_iter();
        let weight_value = it
            .next()
            .ok_or_else(|| Error::Invalid("weighted presentation item lost weight".into()))?;
        let weight = expect_f64(weight_value, "weight")?;
        let center_value = it
            .next()
            .ok_or_else(|| Error::Invalid("weighted presentation item lost center".into()))?;
        let json::Value::Array(coords) = center_value else {
            return Err(Error::Invalid(format!(
                "presentation[{i}].center must be a vector"
            )));
        };
        let mut center = Vec::with_capacity(coords.len());
        for c in coords {
            center.push(expect_f64(c, "vector component")?);
        }
        let variance_value = it
            .next()
            .ok_or_else(|| Error::Invalid("weighted presentation item lost variance".into()))?;
        let variance = expect_f64(variance_value, "variance")?;
        out.push(InputAtom {
            weight,
            center,
            variance,
        });
    }
    Ok(out)
}

pub fn summary_json(summary: &Summary) -> String {
    let mut out = String::with_capacity(640);
    out.push('{');
    out.push_str("\"steps_seen\":");
    out.push_str(&summary.steps_seen.to_string());
    out.push_str(",\"dimension\":");
    out.push_str(&summary.dimension.to_string());
    out.push_str(",\"scalar\":");
    sink_quote(&mut out, summary.scalar);
    out.push_str(",\"memory\":");
    push_float(&mut out, summary.memory);
    out.push_str(",\"eta\":");
    push_float(&mut out, summary.eta);
    out.push_str(",\"mode\":");
    sink_quote(&mut out, summary.mode.as_str());
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
    sink_quote(&mut out, &summary.budget);
    out.push_str(",\"budget_units\":");
    out.push_str(&summary.budget_units.to_string());
    out.push_str(",\"budget_margin_units\":");
    out.push_str(&summary.budget_margin_units.to_string());
    out.push_str(",\"is_solvent\":");
    out.push_str(if summary.is_solvent { "true" } else { "false" });
    out.push('}');
    out
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

fn write_f64_vec(out: &mut impl JsonSink, values: &[f64]) {
    out.push('[');
    for (i, x) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_float(out, *x);
    }
    out.push(']');
}

fn write_output_presentation(out: &mut impl JsonSink, presentation: &OutputPresentation) {
    out.push('[');
    for (i, atom) in presentation.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        push_float(out, atom.weight);
        out.push(',');
        write_f64_vec(out, &atom.center);
        out.push(',');
        push_float(out, atom.variance);
        out.push(']');
    }
    out.push(']');
}

fn write_output_family(out: &mut impl JsonSink, family: &[OutputPresentation]) {
    out.push('[');
    for (i, presentation) in family.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_output_presentation(out, presentation);
    }
    out.push(']');
}

fn write_step_report_sink(out: &mut impl JsonSink, report: &StepReport) {
    out.push('{');
    out.push_str("\"step_index\":");
    out.push_str(&report.step_index.to_string());
    out.push_str(",\"readout\":{");
    match &report.readout {
        Readout::Geometry { present } => {
            out.push_str("\"present\":");
            write_output_family(out, present);
        }
        Readout::Predictive { present, future } => {
            out.push_str("\"present\":");
            write_output_family(out, present);
            out.push_str(",\"future\":");
            write_output_family(out, future);
        }
    }
    out.push('}');
    out.push_str(",\"transformations\":[");
    for (i, t) in report.transformations.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_transformation(out, t);
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
        write_layer_report(out, r);
    }
    out.push_str("],\"temporal_reports\":[");
    for (i, r) in report.temporal_reports.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_layer_report(out, r);
    }
    out.push_str("]}");
}

pub fn step_report_json(report: &StepReport) -> String {
    let mut out = String::with_capacity(768);
    write_step_report_sink(&mut out, report);
    out
}

/// Stream a StepReport directly to a writer.  The network is borrowed only;
/// an I/O failure cannot mutate cognitive state.
pub fn write_step_report_json<W: IoWrite>(writer: &mut W, report: &StepReport) -> Result<()> {
    let mut sink = IoJsonSink::new(writer);
    write_step_report_sink(&mut sink, report);
    sink.finish()
}

fn write_transformation(out: &mut impl JsonSink, t: &Transformation) {
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
            sink_quote(out, space.as_str());
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

fn write_layer_report(out: &mut impl JsonSink, r: &LayerReport) {
    out.push('{');
    macro_rules! int_field {
        ($name:literal, $value:expr) => {{
            out.push_str(concat!("\"", $name, "\":"));
            out.push_str(&$value.to_string());
        }};
    }
    out.push_str("\"phase\":");
    sink_quote(out, r.phase.as_str());
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
        Auxein::new(1, 10.0, 1.0, Budget::kernels("100")).unwrap()
    }

    fn make_predictive64() -> Auxein<f64> {
        Auxein::new_with_mode(1, 10.0, 1.0, Mode::Predictive, Budget::kernels("100")).unwrap()
    }

    #[test]
    fn packing() {
        let n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("0")).unwrap();
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
            assert_eq!(report.readout.present().len(), 1);
            let atoms = &report.readout.present()[0];
            assert_eq!(atoms.len(), 1);
            assert_eq!(atoms[0].weight, 1.0);
            assert_eq!(atoms[0].center, vec![x]);
            assert_eq!(atoms[0].variance, 0.0);
        }
    }

    #[test]
    fn f64_support_underflow_is_not_cognitive_death() {
        let mut n = Auxein::<f64>::new(1, 0.25, 1.0, Budget::kernels("1000")).unwrap();
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
        assert_eq!(r3.readout.present().len(), 1);
        assert_eq!(r3.readout.present()[0].len(), 1);
        assert_eq!(r3.readout.present()[0][0].center, vec![2.0]);
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
        let restored = Auxein::<f64>::from_json(&after, Budget::units(n.budget_units())).unwrap();
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
        let mut a = Auxein::<f32>::new(3, 10.0, 1.0, Budget::kernels("100")).unwrap();
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
        let n = Auxein::<f32>::new(1, 10.1, 0.7, Budget::kernels("100")).unwrap();
        assert_ne!(n.memory(), 10.1);
        let state = n.export_json();
        let restored = Auxein::<f32>::from_json(&state, Budget::units(n.budget_units())).unwrap();
        assert_eq!(restored.export_json(), state);
    }

    #[test]
    fn f32_projected_seed_is_revalidated_before_persistence() {
        let state = r#"{"format_version":5,"dimension":2,"scalar":"f32","memory":1.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0,1.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f32>::from_json(state, Budget::kernels("100")).unwrap();

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
            Auxein::<f32>::from_json(&persisted, Budget::units(n.budget_units())).unwrap();
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
        let mut small = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("1")).unwrap();
        let report = small.step(&[vec![-2.0], vec![2.0]], false).unwrap();
        assert_eq!(small.layers()[0].sigma().len(), 0);
        assert!(report
            .transformations
            .iter()
            .any(|t| matches!(t, Transformation::GrowthReject { .. })));

        let mut roomy = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("2")).unwrap();
        roomy.step(&[vec![-2.0], vec![2.0]], false).unwrap();
        assert_eq!(roomy.layers()[0].sigma().len(), 2);
    }

    #[test]
    fn context_frontier_is_in_growth_transaction() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut roomy = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
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
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":3.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
        let report = n.step(&[vec![3.0]], true).unwrap();
        let masses = &report.layer_reports[0].cell_responsibility_mass;
        assert_eq!(report.readout.present().len(), 1);
        assert_eq!(report.readout.present()[0].len(), 2);
        assert!((stable_sum(masses.iter().copied()) - 1.0).abs() < 1e-15);
        assert!((masses[0] - 5.0 / 29.0).abs() < 1e-15);
        assert!((masses[1] - 24.0 / 29.0).abs() < 1e-15);
    }

    #[test]
    fn forced_solvency_destroys_work_then_knowledge() {
        let mut n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("20")).unwrap();
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
    fn large_multiwinner_readout_closes_mass_without_false_overflow() {
        let mut cells = Vec::new();
        let mut concerned = Vec::new();
        for i in 0..4096usize {
            let center = [1.0 + i as f64 * 1e-9];
            cells.push(Kernel::<f64>::new(1.0, &center, 10.0).unwrap());
            concerned.push((i, 3.0 + i as f64 * 1e-12));
        }
        let mut contributions = Vec::new();
        append_gain_weighted_knowledge(&cells, &concerned, 1.0, &mut contributions);
        coalesce_knowledge_contributions(&cells, &mut contributions);
        let knowledge = build_output_knowledge(&cells, &contributions);
        let output = complete_output_presentation(knowledge, 1).unwrap();
        let total = stable_sum(output.iter().map(|atom| atom.weight));
        assert!(total > 0.0 && total <= 1.0, "total={total:.17e}");
        assert!(output.iter().all(|atom| atom.weight > 0.0));
    }

    #[test]
    fn strict_state_rejects_unrepresentable_f32_config() {
        let bad = r#"{"format_version":5,"dimension":1,"scalar":"f32","memory":10.1,"eta":0.7,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[]}]}"#;
        assert!(Auxein::<f32>::from_json(bad, Budget::kernels("10")).is_err());
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
        targets.reset(1, 1, true).unwrap();
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
            general_targets
                .reset(original.len(), x.len(), true)
                .unwrap();
            for &(index, weight) in &responsibilities {
                general_targets.add_atom(index, 0, weight);
            }
            let mut general = original.clone();
            general_targets
                .apply_population(&mut general, &presentation, beta, lambda)
                .unwrap();

            let mut single_targets = Targets::default();
            single_targets
                .reset(original.len(), x.len(), false)
                .unwrap();
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
        let state_a = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let state_b = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":100.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut a = Auxein::<f64>::from_json(state_a, Budget::kernels("100")).unwrap();
        let mut b = Auxein::<f64>::from_json(state_b, Budget::kernels("100")).unwrap();
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
        assert!(
            (ra.layer_reports[0].context_center.as_ref().unwrap()[0] - 21.0 / 13.0).abs() < 1e-15
        );
        assert!((ra.layer_reports[0].context_variance.unwrap() - 520.0 / 2197.0).abs() < 1e-15);
    }

    #[test]
    fn context_mass_is_recognised_input_mass_without_duplication() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
        let r = n.step(&[vec![1.0], vec![3.0], vec![-10.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.input_mass, 1.0);
        assert_eq!(layer.output_mass, 2.0 / 3.0);
        assert_eq!(layer.context_center, Some(vec![2.0]));
        assert_eq!(layer.context_variance, Some(1.0));

        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
        let r = n.step(&[vec![3.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.output_mass, 1.0);
        assert!((layer.context_center.as_ref().unwrap()[0] - 21.0 / 13.0).abs() < 1e-15);
        assert!((layer.context_variance.unwrap() - 520.0 / 2197.0).abs() < 1e-15);
    }

    #[test]
    fn singleton_and_zero_center_contexts_are_vertical_silence() {
        let singleton = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[2.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(singleton, Budget::kernels("100")).unwrap();
        let r = n.step(&[vec![2.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![2.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(0.0));
        assert!(!r.layer_reports[0].context_emitted);

        let symmetric = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-1.0],"V":0.0},{"W":1.0,"C":[1.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(symmetric, Budget::kernels("100")).unwrap();
        let r = n.step(&[vec![-1.0], vec![1.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![0.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(1.0));
        assert!(!r.layer_reports[0].context_emitted);
    }

    #[test]
    fn perfect_pair_emits_one_context_and_stops_after_l1_learns_it() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("1000")).unwrap();
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
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":1.0},{"W":1.0,"C":[2.0],"V":1.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("1000")).unwrap();
        for _ in 0..40 {
            n.step(&[vec![1.5]], false).unwrap();
        }
        assert!(n.layers().len() <= 2);
        if n.layers().len() == 2 {
            assert!(n.layers()[1].cells().len() <= 1);
        }
    }

    #[test]
    fn lazy_decay_preserves_cognitive_trajectory() {
        fn close_support<S: Scalar>(a: f64, b: f64) -> bool {
            let scale = a.abs().max(b.abs()).max(S::min_positive().to_f64());
            let tolerance = if S::NAME == "f32" { 2.0e-5 } else { 2.0e-12 };
            (a - b).abs() <= tolerance * scale
        }

        fn assert_presentations_close<S: Scalar>(
            left: &[OutputPresentation],
            right: &[OutputPresentation],
        ) {
            assert_eq!(left.len(), right.len());
            for (a, b) in left.iter().zip(right) {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b) {
                    assert!(close_support::<S>(x.weight, y.weight));
                    assert!(close_support::<S>(x.variance, y.variance));
                    assert_eq!(x.center.len(), y.center.len());
                    for (&xc, &yc) in x.center.iter().zip(&y.center) {
                        assert!(close_support::<S>(xc, yc));
                    }
                }
            }
        }

        fn assert_report_causally_equal<S: Scalar>(a: &StepReport, b: &StepReport) {
            assert_eq!(a.step_index, b.step_index);
            assert_eq!(a.transformations, b.transformations);
            assert_eq!(a.maintenance_open_units, b.maintenance_open_units);
            assert_eq!(a.maintenance_units, b.maintenance_units);
            assert_eq!(a.budget_units, b.budget_units);
            assert_presentations_close::<S>(a.readout.present(), b.readout.present());
            assert_presentations_close::<S>(a.readout.future(), b.readout.future());
            assert_eq!(a.layer_reports.len(), b.layer_reports.len());
            assert_eq!(a.temporal_reports.len(), b.temporal_reports.len());
            for (x, y) in a
                .layer_reports
                .iter()
                .chain(&a.temporal_reports)
                .zip(b.layer_reports.iter().chain(&b.temporal_reports))
            {
                assert_eq!(x.phase, y.phase);
                assert_eq!(x.layer_index, y.layer_index);
                assert_eq!(x.input_atom_count, y.input_atom_count);
                assert_eq!(x.unknown_atom_count, y.unknown_atom_count);
                assert_eq!(x.recognised_atom_count, y.recognised_atom_count);
                assert_eq!(x.cell_count_before, y.cell_count_before);
                assert_eq!(x.cell_count_after, y.cell_count_after);
                assert_eq!(x.sigma_count_before, y.sigma_count_before);
                assert_eq!(x.sigma_count_after, y.sigma_count_after);
                assert_eq!(x.promoted, y.promoted);
                assert_eq!(x.seed_requests, y.seed_requests);
                assert_eq!(x.context_emitted, y.context_emitted);
                assert_eq!(x.output_atom_count, y.output_atom_count);
                assert_eq!(x.recognition_count, y.recognition_count);
                assert_eq!(x.present_atom_count, y.present_atom_count);
                assert!(close_support::<S>(x.input_mass, y.input_mass));
                assert!(close_support::<S>(x.output_mass, y.output_mass));
                assert!(close_support::<S>(x.knowledge_mass, y.knowledge_mass));
                assert!(close_support::<S>(x.present_mass, y.present_mass));
                assert_eq!(
                    x.cell_responsibility_mass.len(),
                    y.cell_responsibility_mass.len()
                );
                for (&r1, &r2) in x
                    .cell_responsibility_mass
                    .iter()
                    .zip(&y.cell_responsibility_mass)
                {
                    assert!(close_support::<S>(r1, r2));
                }
                match (&x.context_center, &y.context_center) {
                    (Some(cx), Some(cy)) => {
                        assert_eq!(cx.len(), cy.len());
                        for (&vx, &vy) in cx.iter().zip(cy) {
                            assert!(close_support::<S>(vx, vy));
                        }
                    }
                    (None, None) => {}
                    _ => panic!("context topology differs"),
                }
                match (x.context_variance, y.context_variance) {
                    (Some(vx), Some(vy)) => assert!(close_support::<S>(vx, vy)),
                    (None, None) => {}
                    _ => panic!("context topology differs"),
                }
            }
        }

        fn run<S: Scalar>() {
            let scalar = S::NAME;
            let state = format!(
                "{{\"format_version\":5,\"dimension\":1,\"scalar\":\"{scalar}\",\"memory\":23.0,\"eta\":1.0,\"mode\":\"geometry\",\"steps_seen\":0,\"layers\":[{{\"sigma\":[],\"cells\":[{{\"W\":1.0,\"C\":[-3.0],\"V\":0.25}},{{\"W\":2.0,\"C\":[1.0],\"V\":0.25}},{{\"W\":3.0,\"C\":[4.0],\"V\":0.25}}]}}]}}"
            );
            let mut lazy = Auxein::<S>::from_json(&state, Budget::kernels("100")).unwrap();
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
                // Readout, transformations, growth/contraction and therefore
                // causal topology must stay identical.  Historical support
                // projection may differ by a few ulps because the lazy path
                // evaluates the same homothety in O(log age).
                assert_report_causally_equal::<S>(&a, &b);

                for layer in &mut eager.layers {
                    let epoch = layer.cell_decay.epoch();
                    let lambda = layer.cell_decay.lambda();
                    for kernel in layer.sigma.iter_mut().chain(&mut layer.cells) {
                        kernel.materialize_weight_at(epoch, lambda);
                    }
                }
                assert_eq!(lazy.layers.len(), eager.layers.len());
                for (left, right) in lazy.layers.iter().zip(&eager.layers) {
                    assert_eq!(left.cells.len(), right.cells.len());
                    assert_eq!(left.sigma.len(), right.sigma.len());
                    for (a, b) in left.cells.iter().zip(&right.cells) {
                        assert_eq!(a.center.len(), b.center.len());
                        for (&x, &y) in a.center.iter().zip(&b.center) {
                            assert!(close_support::<S>(x.to_f64(), y.to_f64()));
                        }
                        assert!(close_support::<S>(a.variance.to_f64(), b.variance.to_f64()));
                        assert!(close_support::<S>(a.weight(), b.weight()));
                    }
                    for (a, b) in left.sigma.iter().zip(&right.sigma) {
                        assert_eq!(a.center.len(), b.center.len());
                        for (&x, &y) in a.center.iter().zip(&b.center) {
                            assert!(close_support::<S>(x.to_f64(), y.to_f64()));
                        }
                        assert!(close_support::<S>(a.variance.to_f64(), b.variance.to_f64()));
                        assert!(close_support::<S>(a.weight(), b.weight()));
                    }
                }
            }
        }
        run::<f64>();
        run::<f32>();
    }

    #[test]
    fn extreme_squared_norm_is_never_nan() {
        assert_eq!(norm2(&[f64::MAX]), f64::INFINITY);
        assert_eq!(norm2(&[f64::from_bits(1)]), 0.0);
        assert!(!norm2(&[1.0, f64::MAX]).is_nan());
    }

    #[test]
    fn distinct_subnormal_centers_keep_positive_context_variance() {
        fn run<S: Scalar>() {
            let a = S::min_positive().to_f64();
            let cells = vec![
                Kernel::<S>::new(1.0, &[a], 0.0).unwrap(),
                Kernel::<S>::new(1.0, &[2.0 * a], 0.0).unwrap(),
            ];
            let context = build_context_kernel(&cells, &[(0, 0.5), (1, 0.5)], 1).unwrap();
            assert!(context.variance > 0.0);
            assert!(context.variance.is_finite());
            let projected = project_kernel::<S>(context.clone())
                .unwrap()
                .variance
                .to_f64();
            eprintln!(
                "scalar={} context_v={:?} projected_v={:?}",
                S::NAME,
                context.variance,
                projected
            );
            assert!(projected > 0.0);
        }
        run::<f64>();
        run::<f32>();
    }

    #[test]
    fn extreme_context_variance_saturates_finitely() {
        let cells = vec![
            Kernel::<f64>::new(1.0, &[-f64::MAX], 0.0).unwrap(),
            Kernel::<f64>::new(1.0, &[f64::MAX], 0.0).unwrap(),
        ];
        let context = build_context_kernel(&cells, &[(0, 0.5), (1, 0.5)], 1).unwrap();
        assert!(context.variance.is_finite());
        assert_eq!(context.variance, f64::MAX);
    }

    #[test]
    fn contraction_value_is_finite_for_extreme_centers() {
        let huge = Kernel::<f64>::new(1.0, &[f64::MAX], f64::MAX).unwrap();
        let tiny = Kernel::<f64>::new(1.0, &[f64::from_bits(1)], 1.0).unwrap();
        for k in [
            Auxein::<f64>::cell_value(&huge),
            Auxein::<f64>::cell_value(&tiny),
        ] {
            assert!(k.is_finite());
            assert!(k > 0.0 && k <= 1.0);
        }
    }

    #[test]
    fn lazy_decay_clock_is_clone_local() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":31.0,"eta":1.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-3.0],"V":0.25},{"W":2.0,"C":[1.0],"V":0.25},{"W":3.0,"C":[4.0],"V":0.25}]}]}"#;
        let mut original = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
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
    fn predictive_packing() {
        let n = Auxein::<f64>::new_with_mode(1, 10.0, 1.0, Mode::Predictive, Budget::kernels("0"))
            .unwrap();
        assert_eq!(n.temporal_kernel_units().unwrap(), 32);
        assert_eq!(n.layer_units().unwrap(), 57);
        assert_eq!(n.min_units().unwrap(), 107);
    }

    #[test]
    fn mode_is_geometry_by_default_and_strict() {
        assert_eq!(Mode::parse("geometry").unwrap(), Mode::Geometry);
        assert_eq!(Mode::parse("predictive").unwrap(), Mode::Predictive);
        assert!(Mode::parse("temporal").is_err());
        assert_eq!(make64().mode(), Mode::Geometry);
    }

    #[test]
    fn weighted_boundary_and_zero_remainder_are_canonical() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":9.0,"C":[2.0],"V":77.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100")).unwrap();
        let input = vec![
            InputAtom::new(0.25, vec![2.0], 0.0),
            InputAtom::new(0.75, vec![99.0], 0.0),
        ];
        let report = n.step_weighted(&input, false).unwrap();
        assert_eq!(report.readout.present().len(), 1);
        let p = &report.readout.present()[0];
        assert_eq!(p.len(), 2);
        assert_eq!(
            p[0],
            OutputAtom {
                weight: 0.75,
                center: vec![0.0],
                variance: 0.0
            }
        );
        assert_eq!(
            p[1],
            OutputAtom {
                weight: 0.25,
                center: vec![2.0],
                variance: 0.0
            }
        );
        assert!((stable_sum(p.iter().map(|a| a.weight)) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn weighted_boundary_rejects_invalid_mass() {
        let mut n = make64();
        let bad = vec![
            InputAtom::new(0.6, vec![1.0], 0.0),
            InputAtom::new(0.6, vec![2.0], 0.0),
        ];
        assert!(n.step_weighted(&bad, false).is_err());
        assert!(n.step_weighted(&[], false).is_err());
    }

    #[test]
    fn gain_weighted_context_matches_concern_not_support() {
        let a = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let b = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1e-200,"C":[1.0],"V":10.0},{"W":1e200,"C":[2.0],"V":10.0}]}]}"#;
        let mut na = Auxein::<f64>::from_json(a, Budget::kernels("100")).unwrap();
        let mut nb = Auxein::<f64>::from_json(b, Budget::kernels("100")).unwrap();
        let ra = na.step(&[vec![3.0]], true).unwrap();
        let rb = nb.step(&[vec![3.0]], true).unwrap();
        assert_ne!(
            ra.layer_reports[0].cell_responsibility_mass,
            rb.layer_reports[0].cell_responsibility_mass
        );
        let ca = ra.layer_reports[0].context_center.as_ref().unwrap()[0];
        let cb = rb.layer_reports[0].context_center.as_ref().unwrap()[0];
        assert!((ca - 21.0 / 13.0).abs() < 1e-15);
        assert_eq!(ca, cb);
        assert_eq!(
            ra.layer_reports[0].context_variance,
            rb.layer_reports[0].context_variance
        );
        assert!((ra.layer_reports[0].context_variance.unwrap() - 520.0 / 2197.0).abs() < 1e-15);
    }

    #[test]
    fn step_is_atomic_and_never_learns_cross_call_transition() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[5.0], 0.0).unwrap(),
        ];
        n.step(&[vec![1.0]], false).unwrap();
        n.step(&[vec![5.0]], false).unwrap();
        assert!(n.layers[0].temporal_sigma.is_empty());
        assert!(n.layers[0].temporal_cells.is_empty());
        assert!(n.layers[0].previous.is_none());
    }

    #[test]
    fn explicit_sequence_learns_adjacent_transition_and_clears_previous() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[5.0], 0.0).unwrap(),
        ];
        n.sequence(&[vec![vec![1.0]], vec![vec![5.0]]], false)
            .unwrap();
        assert_eq!(n.layers[0].temporal_sigma.len(), 1);
        assert_eq!(
            scalar_vec_to_f64(&n.layers[0].temporal_sigma[0].center),
            vec![1.0, 5.0]
        );
        assert!(n.layers[0].previous.is_none());
    }

    #[test]
    fn sequence_boundary_prevents_cross_sequence_transition() {
        let mut n = make_predictive64();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[-10.0], 0.0).unwrap(),
            Kernel::new(1.0, &[-1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[10.0], 0.0).unwrap(),
        ];
        n.sequence(&[vec![vec![1.0]], vec![vec![10.0]]], false)
            .unwrap();
        n.sequence(&[vec![vec![-1.0]], vec![vec![-10.0]]], false)
            .unwrap();
        let centers: Vec<Vec<f64>> = n.layers[0]
            .temporal_sigma
            .iter()
            .chain(n.layers[0].temporal_cells.iter())
            .map(|k| scalar_vec_to_f64(&k.center))
            .collect();
        assert!(centers.contains(&vec![1.0, 10.0]), "centers={centers:?}");
        assert!(centers.contains(&vec![-1.0, -10.0]), "centers={centers:?}");
        assert!(!centers.contains(&vec![10.0, -1.0]), "centers={centers:?}");
    }

    #[test]
    fn singleton_can_predict_but_cannot_leave_previous() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 2.0], 0.0).unwrap()];
        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.future().len(), 1);
        assert_eq!(
            report.readout.future()[0],
            vec![OutputAtom {
                weight: 1.0,
                center: vec![2.0],
                variance: 0.0
            }]
        );
        assert!(n.layers[0].previous.is_none());
    }

    #[test]
    fn prediction_preserves_current_context_mass_and_branches_without_ranking() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[2.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[2.0, 7.0], 0.0).unwrap(),
            Kernel::new(999.0, &[2.0, 8.0], 77.0).unwrap(),
        ];
        let input = vec![
            InputAtom::new(0.25, vec![2.0], 0.0),
            InputAtom::new(0.75, vec![99.0], 0.0),
        ];
        let report = n.step_weighted(&input, false).unwrap();
        assert_eq!(report.readout.future().len(), 2);
        for p in report.readout.future() {
            assert_eq!(p.len(), 2);
            assert_eq!(p[0].weight, 0.75);
            assert_eq!(p[0].center, vec![0.0]);
            assert_eq!(p[1].weight, 0.25);
            assert_eq!(p[1].variance, 0.0);
        }
        let targets: Vec<f64> = report
            .readout
            .future()
            .iter()
            .map(|p| p[1].center[0])
            .collect();
        assert_eq!(targets, vec![7.0, 8.0]);
    }

    #[test]
    fn predictive_relative_gain_weights_branches_independently() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[2.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[2.0, 7.0], 0.0).unwrap(),
            Kernel::new(999.0, &[3.9, 8.0], 77.0).unwrap(),
        ];
        let input = vec![
            InputAtom::new(0.25, vec![2.0], 0.0),
            InputAtom::new(0.75, vec![99.0], 0.0),
        ];
        let report = n.step_weighted(&input, false).unwrap();
        assert_eq!(report.readout.future().len(), 2);

        let mut seen = Vec::new();
        for presentation in report.readout.future() {
            let directional = presentation
                .iter()
                .find(|atom| atom.center[0] != 0.0)
                .unwrap();
            seen.push((directional.center[0], directional.weight));
        }
        seen.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(seen[0], (7.0, 0.25));
        assert!((seen[1].0 - 8.0).abs() < f64::EPSILON);
        assert!((seen[1].1 - 0.024375).abs() < 1e-15);
    }

    #[test]
    fn predictive_same_target_uses_max_relative_gain() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[2.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[3.9, 7.0], 0.0).unwrap(),
            Kernel::new(1.0, &[2.0, 7.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.9, 7.0], 0.0).unwrap(),
        ];
        let input = vec![
            InputAtom::new(0.25, vec![2.0], 0.0),
            InputAtom::new(0.75, vec![99.0], 0.0),
        ];
        let report = n.step_weighted(&input, false).unwrap();
        assert_eq!(report.readout.future().len(), 1);
        let directional = report.readout.future()[0]
            .iter()
            .find(|atom| atom.center[0] != 0.0)
            .unwrap();
        assert_eq!(directional.center, vec![7.0]);
        assert_eq!(directional.weight, 0.25);
    }

    #[test]
    fn predictive_relative_gain_is_scale_invariant_at_extremes() {
        for scale in [1.0, 1e-158, 1e-200, 1e158, 1e200] {
            let current = [2.0 * scale];
            let source = [3.9 * scale];
            let current2 = norm2(&current);
            let gamma = point_relative_gain_scalar(&current, current2, true, &source).unwrap();
            assert!(
                (gamma - 0.0975).abs() < 1e-14,
                "scale={scale} gamma={gamma}"
            );
        }
    }

    #[test]
    fn zero_target_is_explicit_and_prediction_is_not_recursive() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![
            Kernel::new(1.0, &[1.0, 0.0], 0.0).unwrap(),
            Kernel::new(1.0, &[1.0, 2.0], 0.0).unwrap(),
            Kernel::new(1.0, &[2.0, 3.0], 0.0).unwrap(),
        ];
        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.future().len(), 2);
        assert!(report.readout.future().contains(&vec![OutputAtom {
            weight: 1.0,
            center: vec![0.0],
            variance: 0.0
        }]));
        assert!(report.readout.future().contains(&vec![OutputAtom {
            weight: 1.0,
            center: vec![2.0],
            variance: 0.0
        }]));
        assert!(!report.readout.future().contains(&vec![OutputAtom {
            weight: 1.0,
            center: vec![3.0],
            variance: 0.0
        }]));
    }

    #[test]
    fn temporal_product_kernel_is_exact_direct_sum_quotient() {
        let n = make_predictive64();
        let a = Kernel64 {
            weight: 0.5,
            center: vec![2.0],
            variance: 1.0,
        };
        let b = Kernel64 {
            weight: 0.25,
            center: vec![7.0],
            variance: 4.0,
        };
        let t = n.temporal_atom(&a, &b).unwrap();
        assert_eq!(t.r, 0.125);
        assert_eq!(t.x.as_ref(), &[2.0, 7.0]);
        assert_eq!(t.variance, 5.0);
    }

    #[test]
    fn eta_zero_advances_previous_inside_sequence_only() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 0.0).unwrap(),
            Kernel::new(1.0, &[3.0], 0.0).unwrap(),
        ];
        n.begin_sequence(false).unwrap();
        n.sequence_step(&[vec![1.0]], false).unwrap();
        assert_eq!(
            scalar_vec_to_f64(&n.layers[0].previous.as_ref().unwrap().center),
            vec![1.0]
        );
        n.sequence_step(&[vec![3.0]], false).unwrap();
        assert_eq!(
            scalar_vec_to_f64(&n.layers[0].previous.as_ref().unwrap().center),
            vec![3.0]
        );
        assert!(n.layers[0].temporal_sigma.is_empty());
        n.end_sequence().unwrap();
        assert!(n.layers[0].previous.is_none());
    }

    #[test]
    fn direct_composition_is_depth_ordered_atomic_geometry() {
        let up_state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]},{"sigma":[],"cells":[{"W":1.0,"C":[2.0],"V":1.0}]}]}"#;
        let mut up = Auxein::<f64>::from_json(up_state, Budget::kernels("1000")).unwrap();
        let family = up
            .step(&[vec![1.0], vec![3.0]], false)
            .unwrap()
            .readout
            .present()
            .to_vec();
        assert_eq!(family.len(), 2);
        let mut down = make_predictive64();
        down.layers[0].cells = vec![
            Kernel::new(1.0, &[1.0], 10.0).unwrap(),
            Kernel::new(1.0, &[2.0], 10.0).unwrap(),
            Kernel::new(1.0, &[3.0], 10.0).unwrap(),
        ];
        let reports = down.consume(&family, false).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(down.layers[0].temporal_sigma.is_empty());
        assert!(down.layers[0].temporal_cells.is_empty());
        assert!(down.layers[0].previous.is_none());
    }

    #[test]
    fn predictive_roundtrip_preserves_private_state_and_atomic_default() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 0.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(1.0, &[1.0, 2.0], 0.0).unwrap()];
        let state = n.export_json();
        assert!(state.contains("\"format_version\":5"));
        let mut restored =
            Auxein::<f64>::from_json(&state, Budget::units(n.budget_units())).unwrap();
        assert_eq!(restored.export_json(), state);
        let report = restored.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.future()[0][0].center, vec![2.0]);
        assert!(restored.layers[0].previous.is_none());
    }

    #[test]
    fn predictive_and_geometry_share_geometric_trajectory() {
        let mut g = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("10000")).unwrap();
        let mut p =
            Auxein::<f64>::new_with_mode(1, 10.0, 1.0, Mode::Predictive, Budget::kernels("10000"))
                .unwrap();
        let sequence = vec![
            vec![vec![1.0]],
            vec![vec![5.0]],
            vec![vec![1.0]],
            vec![vec![5.0]],
            vec![vec![1.0], vec![5.0]],
        ];
        for _ in 0..4 {
            for presentation in &sequence {
                g.step(presentation, false).unwrap();
            }
            p.sequence(&sequence, false).unwrap();
        }
        assert_eq!(g.layers.len(), p.layers.len());
        for (gl, pl) in g.layers.iter().zip(&p.layers) {
            assert_eq!(gl.cells, pl.cells);
            assert_eq!(gl.sigma, pl.sigma);
        }
    }

    #[test]
    fn temporal_population_does_not_age_without_temporal_presentation() {
        let mut n =
            Auxein::<f64>::new_with_mode(1, 10.0, 1.0, Mode::Predictive, Budget::kernels("100"))
                .unwrap();
        n.layers[0].cells = vec![Kernel::new(1.0, &[1.0], 0.0).unwrap()];
        n.layers[0].temporal_cells = vec![Kernel::new(2.0, &[1.0, 1.0], 0.5).unwrap()];
        let before = n.layers[0].temporal_cells[0].clone();
        n.sequence(&[vec![vec![99.0]], vec![vec![99.0]]], false)
            .unwrap();
        assert_eq!(n.layers[0].temporal_cells[0], before);
    }

    #[test]
    fn steps_seen_saturates_without_stopping_cognition() {
        let state = r#"{"format_version":5,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"mode":"geometry","steps_seen":18446744073709551615,"layers":[{"sigma":[],"cells":[]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("10")).unwrap();
        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.step_index, u64::MAX);
        assert_eq!(n.steps_seen(), u64::MAX);
        let report2 = n.step(&[vec![2.0]], false).unwrap();
        assert_eq!(report2.step_index, u64::MAX);
        assert_eq!(n.steps_seen(), u64::MAX);
    }

    #[test]
    fn decay_clock_rollover_is_transparent() {
        let mut n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("100")).unwrap();
        let clock = n.layers[0].cell_decay.clone();
        let mut cell = Kernel::new(1.0, &[1.0], 0.0).unwrap();
        cell.bind_decay_clock(clock.clone(), 0);
        n.layers[0].cells = vec![cell];
        clock.epoch.store(u64::MAX, AtomicOrdering::Relaxed);
        let report = n.step(&[vec![1.0]], false).unwrap();
        assert_eq!(report.readout.present().len(), 1);
        assert_eq!(clock.epoch(), 1);
        assert!(n.layers[0].cells[0].weight().is_finite());
        assert!(n.layers[0].cells[0].weight() > 0.0);
    }

    #[test]
    fn zero_lambda_is_a_normal_long_running_case() {
        let mut n = Auxein::<f64>::new(1, f64::MIN_POSITIVE, 1.0, Budget::kernels("100")).unwrap();
        assert_eq!(n.lambda(), 0.0);
        let clock = n.layers[0].cell_decay.clone();
        let mut cell = Kernel::new(1.0, &[1.0], 0.0).unwrap();
        cell.bind_decay_clock(clock, 0);
        n.layers[0].cells = vec![cell];
        n.step(&[vec![100.0]], false).unwrap();
        assert_eq!(n.layers[0].cells[0].weight(), f64::from_bits(1));
    }

    #[test]
    fn unrepresentable_f32_learning_input_is_rejected_before_mutation() {
        let mut n = Auxein::<f32>::new(1, 10.0, 1.0, Budget::kernels("100")).unwrap();
        let before = n.export_json();
        assert!(n.step(&[vec![f64::MAX]], false).is_err());
        assert_eq!(n.export_json(), before);

        let input = [InputAtom {
            weight: 1.0,
            center: vec![1.0],
            variance: f64::MAX,
        }];
        assert!(n.step_weighted(&input, false).is_err());
        assert_eq!(n.export_json(), before);
    }

    #[test]
    fn streaming_state_matches_string_serializer() {
        let mut n = make_predictive64();
        n.sequence(
            &[
                vec![vec![1.0]],
                vec![vec![2.0]],
                vec![vec![1.0]],
                vec![vec![2.0]],
            ],
            false,
        )
        .unwrap();
        let expected = n.export_json();
        let mut bytes = Vec::new();
        n.write_json(&mut bytes).unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), expected);
    }

    #[test]
    fn streaming_report_matches_string_serializer() {
        let mut n = make64();
        let report = n.step(&[vec![1.0], vec![2.0]], true).unwrap();
        let expected = step_report_json(&report);
        let mut bytes = Vec::new();
        write_step_report_json(&mut bytes, &report).unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), expected);
    }

    struct FailWriter {
        remaining: usize,
    }

    impl std::io::Write for FailWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "injected"));
            }
            let written = buf.len().min(self.remaining);
            self.remaining -= written;
            Ok(written)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn streaming_io_failure_does_not_mutate_state() {
        let mut n = make64();
        n.step(&[vec![1.0]], false).unwrap();
        let before = n.export_json();
        let mut writer = FailWriter { remaining: 17 };
        assert!(matches!(n.write_json(&mut writer), Err(Error::Io(_))));
        assert_eq!(n.export_json(), before);
    }

    #[test]
    fn report_io_failure_is_structured() {
        let mut n = make64();
        let report = n.step(&[vec![1.0]], true).unwrap();
        let mut writer = FailWriter { remaining: 11 };
        assert!(matches!(
            write_step_report_json(&mut writer, &report),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn transient_scratch_can_be_compacted_without_cognitive_change() {
        let mut n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("10000")).unwrap();
        let clock = n.layers[0].cell_decay.clone();
        n.layers[0].cells = (1..=512)
            .map(|i| {
                let mut kernel = Kernel::new(1.0, &[i as f64], 1_000_000.0).unwrap();
                kernel.bind_decay_clock(clock.clone(), 0);
                kernel
            })
            .collect();
        n.step(&[vec![1000.0]], false).unwrap();
        assert!(n.transient_memory_capacity_bytes() > 0);
        let before = n.export_json();
        n.compact_transient_memory();
        assert_eq!(n.transient_memory_capacity_bytes(), 0);
        assert_eq!(n.export_json(), before);
    }

    #[test]
    fn hostile_json_nesting_is_bounded() {
        let depth = 24usize; // parser bound is intentionally below this adversarial depth
        let text = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        assert!(json::parse(&text).is_err());
        assert!(parse_presentation_json(&text).is_err());
        assert!(parse_weighted_presentation_json(&text).is_err());
    }

    #[test]
    fn malformed_ascii_json_never_panics() {
        let alphabet = b"[]{}:,\\\"-+.eE0123456789truefalsenull abcXYZ";
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for len in 0..256usize {
            for _ in 0..16 {
                let mut text = String::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    text.push(alphabet[(state as usize) % alphabet.len()] as char);
                }
                let result = std::panic::catch_unwind(|| {
                    let _ = json::parse(&text);
                    let _ = parse_presentation_json(&text);
                    let _ = parse_weighted_presentation_json(&text);
                    let _ = Auxein::<f64>::from_json(&text, Budget::kernels("10"));
                });
                assert!(result.is_ok(), "parser panicked on {text:?}");
            }
        }
    }
}
