#![forbid(unsafe_code)]

//! Auxein v0.2.0 production core.
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

pub const FORMAT_VERSION: u64 = 2;

pub type Result<T> = std::result::Result<T, Error>;

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
}

impl<S: Scalar> PartialEq for Layer<S> {
    fn eq(&self, other: &Self) -> bool {
        self.sigma == other.sigma && self.cells == other.cells
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
        Self {
            sigma: self.sigma.clone(),
            cells,
            cell_decay: clock,
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
        layer: usize,
        count: usize,
    },
    GrowthCommit {
        seeds: usize,
        layer_created: bool,
        units: u64,
    },
    GrowthReject {
        seeds: usize,
        layer_requested: bool,
        units: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerReport {
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
    pub readout: Vec<Recognition>,
    pub transformations: Vec<Transformation>,
    pub maintenance_open_units: u64,
    pub maintenance_units: u64,
    pub budget_units: u64,
    pub layer_reports: Vec<LayerReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub steps_seen: u64,
    pub dimension: usize,
    pub universe: String,
    pub scalar: &'static str,
    pub memory: f64,
    pub eta: f64,
    pub chi: f64,
    pub alpha: f64,
    pub effective_alpha: f64,
    pub layer_count: usize,
    pub cells_per_layer: Vec<usize>,
    pub sigma_per_layer: Vec<usize>,
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

#[derive(Clone, Debug, Default)]
struct Targets {
    weights: Vec<f64>,
    centers: Vec<f64>,
    variances: Vec<f64>,
    touched: Vec<usize>,
    changed: Vec<usize>,
    dimension: usize,
}

impl Targets {
    fn reset(&mut self, count: usize, dimension: usize, need_centers: bool) {
        for index in self.touched.drain(..) {
            self.weights[index] = 0.0;
        }
        self.dimension = dimension;
        if self.weights.len() < count {
            self.weights.resize(count, 0.0);
            self.variances.resize(count, 0.0);
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

    fn add_atom(&mut self, index: usize, x: &[f64], variance: f64, weight: f64) {
        if weight <= 0.0 {
            return;
        }
        let old_w = self.weights[index];
        let start = index * self.dimension;
        let end = start + self.dimension;
        if old_w == 0.0 {
            self.touched.push(index);
            self.weights[index] = weight;
            self.centers[start..end].copy_from_slice(x);
            self.variances[index] = variance;
            return;
        }
        let total = old_w + weight;
        let ratio = weight / total;
        let mut delta_sum = 0.0;
        let mut delta_correction = 0.0;
        for (old, &new) in self.centers[start..end].iter_mut().zip(x) {
            let d = new - *old;
            compensated_add(&mut delta_sum, &mut delta_correction, d * d);
            *old = structural_zero(*old + ratio * d);
        }
        let delta2 = compensated_finish(delta_sum, delta_correction);
        self.variances[index] = (old_w * self.variances[index] + weight * variance) / total
            + (old_w * weight / (total * total)) * delta2;
        self.weights[index] = total;
    }

    fn apply_population<S: Scalar>(
        &mut self,
        kernels: &mut [Kernel<S>],
        beta: f64,
        lambda: f64,
    ) -> Result<()> {
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
        beta: f64,
        lambda: f64,
        epoch: u32,
    ) -> Result<()> {
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
            + (a * b / (total * total)) * delta2;
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
            steps_seen: 0,
            layers: vec![Layer {
                sigma: Vec::new(),
                cells: Vec::new(),
                cell_decay: Arc::new(DecayClock::new(0, 1.0)),
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

    pub fn network_units(&self) -> Result<u64> {
        33u64
            .checked_add(2 * S::BYTES)
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))
    }

    pub fn min_units(&self) -> Result<u64> {
        self.network_units()?
            .checked_add(16)
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
        }
        self.eta = eta;
        self.refresh_clock();
        for layer in &mut self.layers {
            for cell in &mut layer.cells {
                cell.decay_epoch = 0;
            }
            layer.cell_decay.reset(self.lambda);
        }
        Ok(())
    }

    pub fn maintenance_units(&self) -> Result<u64> {
        let payloads: u64 = self
            .layers
            .iter()
            .map(|l| (l.sigma.len() + l.cells.len()) as u64)
            .sum();
        self.network_units()?
            .checked_add(
                16u64
                    .checked_mul(self.layers.len() as u64)
                    .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?,
            )
            .and_then(|n| n.checked_add(payloads.checked_mul(self.kernel_units().ok()?)?))
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))
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
        let mut readout = Vec::new();
        let mut readout_layers = 0usize;
        let mut all_seed_requests: Vec<(usize, Kernel64)> = Vec::new();
        let mut layer_reports = Vec::new();
        let mut frontier_requested = false;

        let mut current = presentation;
        for layer_index in 0..layer_count_start {
            if current.is_empty() {
                break;
            }
            let result = self.process_layer(layer_index, &current, detailed_report)?;
            if !result.readout.is_empty() {
                readout_layers += 1;
                readout.extend(result.readout);
            }
            transformations.extend(result.transformations);
            all_seed_requests.extend(
                result
                    .seed_requests
                    .into_iter()
                    .map(|seed| (layer_index, seed)),
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

        let seeds = all_seed_requests;
        let seed_count = seeds.len();
        let need_frontier = frontier_requested;
        let growth_cost = (seed_count as u64)
            .checked_mul(self.kernel_units()?)
            .and_then(|n| n.checked_add(if need_frontier { 16 } else { 0 }))
            .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;

        if growth_cost > 0 {
            let payable = self
                .maintenance_units()?
                .checked_add(growth_cost)
                .is_some_and(|n| n <= self.budget_units);
            if payable {
                let mut seeds = seeds.into_iter().peekable();
                while let Some((layer_index, seed)) = seeds.next() {
                    let mut additions = vec![project_kernel::<S>(seed)?];
                    while seeds
                        .peek()
                        .is_some_and(|(next_layer, _)| *next_layer == layer_index)
                    {
                        let (_, seed) = seeds.next().unwrap();
                        additions.push(project_kernel::<S>(seed)?);
                    }
                    additions.append(&mut self.layers[layer_index].sigma);
                    self.layers[layer_index].sigma = coalesce_projected(additions)?;
                }
                if need_frontier {
                    self.layers.push(Layer {
                        sigma: Vec::new(),
                        cells: Vec::new(),
                        cell_decay: Arc::new(DecayClock::new(0, self.lambda)),
                    });
                }
                transformations.push(Transformation::GrowthCommit {
                    seeds: seed_count,
                    layer_created: need_frontier,
                    units: growth_cost,
                });
            } else {
                transformations.push(Transformation::GrowthReject {
                    seeds: seed_count,
                    layer_requested: need_frontier,
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
            dedup_single_layer_recognitions(&mut readout);
        } else {
            sort_dedup_recognitions(&mut readout);
        }
        Ok(StepReport {
            step_index: self.steps_seen - 1,
            readout,
            transformations,
            maintenance_open_units: maintenance_open,
            maintenance_units: maintenance_end,
            budget_units: self.budget_units,
            layer_reports,
        })
    }

    fn presentation(&self, value: &[Vec<f64>]) -> Result<Vec<Atom>> {
        if value.is_empty() {
            return Err(Error::Invalid(
                "external presentation must be a nonempty sequence of vectors".into(),
            ));
        }
        let mass = 1.0 / value.len() as f64;
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
                r: mass,
                variance: 0.0,
                norm2: compensated_finish(norm_sum, norm_correction),
                zero,
            });
        }
        Ok(coalesce_atoms(atoms))
    }

    fn process_layer(
        &mut self,
        layer_index: usize,
        presentation: &[Atom],
        detailed: bool,
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
            targets.reset(old_cells.len(), self.dimension, !point_single);
        }
        let mut cell_received = detailed.then(|| vec![0.0; old_cells.len()]);
        let mut readout = Vec::new();
        let mut context: Option<Kernel64> = None;
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
                        targets.add_atom(ci, &atom.x, atom.variance, rho);
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
                merge_context_point(&mut context, center, share);
                previous = Some(center);
            }
        }

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

        targets.reset(old_sigma.len(), self.dimension, !point_single);
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
                        targets.add_atom(si, &atom.x, atom.variance, tau);
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
            targets.apply_population(&mut updated_sigma, self.beta, self.lambda)?;
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
            readout,
            seed_requests: admissible_seeds,
            transformations,
            report,
        })
    }

    fn force_solvency(&mut self, transformations: &mut Vec<Transformation>) -> Result<()> {
        if self.maintenance_units()? <= self.budget_units {
            return Ok(());
        }

        let removed_sigma: usize = self.layers.iter().map(|l| l.sigma.len()).sum();
        if removed_sigma > 0 {
            for layer in &mut self.layers {
                layer.sigma.clear();
            }
            transformations.push(Transformation::ClearSigma {
                count: removed_sigma,
            });
        }

        let mut trimmed = 0;
        while self.layers.len() > 1 && self.layers.last().is_some_and(|l| l.cells.is_empty()) {
            self.layers.pop();
            trimmed += 1;
        }
        if trimmed > 0 {
            transformations.push(Transformation::TrimLayers { count: trimmed });
        }
        if self.maintenance_units()? <= self.budget_units {
            return Ok(());
        }

        let mut valued: Vec<(f64, usize, usize)> = Vec::new();
        let mut counts: Vec<usize> = self.layers.iter().map(|l| l.cells.len()).collect();
        let mut total_cells = 0usize;
        for (li, layer) in self.layers.iter().enumerate() {
            for (ci, cell) in layer.cells.iter().enumerate() {
                valued.push((Self::cell_value(cell), li, ci));
                total_cells += 1;
            }
        }
        if valued.is_empty() {
            return Err(Error::Inexecutable(
                "minimal Auxein state exceeds the execution budget".into(),
            ));
        }
        valued.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut active_layers = counts.len();
        let mut cutoff = None;
        let mut removed_cells = 0usize;
        let mut waves = 0usize;
        let mut position = 0usize;
        while position < valued.len() {
            let k = valued[position].0;
            let mut stop = position;
            while stop < valued.len() && valued[stop].0 == k {
                counts[valued[stop].1] -= 1;
                total_cells -= 1;
                removed_cells += 1;
                stop += 1;
            }
            waves += 1;
            while active_layers > 1 && counts[active_layers - 1] == 0 {
                active_layers -= 1;
            }
            let simulated = self
                .network_units()?
                .checked_add(16 * active_layers as u64)
                .and_then(|n| {
                    n.checked_add((total_cells as u64).checked_mul(self.kernel_units().ok()?)?)
                })
                .ok_or_else(|| Error::Invalid("material accounting overflow".into()))?;
            cutoff = Some(k);
            if simulated <= self.budget_units {
                break;
            }
            position = stop;
        }
        let Some(cutoff) = cutoff else {
            return Err(Error::Inexecutable(
                "minimal Auxein state exceeds the execution budget".into(),
            ));
        };

        for layer in &mut self.layers {
            layer.cells.retain(|cell| Self::cell_value(cell) > cutoff);
        }
        let mut trimmed_after = 0;
        while self.layers.len() > 1 && self.layers.last().is_some_and(|l| l.cells.is_empty()) {
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
            chi: self.chi,
            alpha: self.alpha,
            effective_alpha: self.beta,
            layer_count: self.layers.len(),
            cells_per_layer: self.layers.iter().map(|l| l.cells.len()).collect(),
            sigma_per_layer: self.layers.iter().map(|l| l.sigma.len()).collect(),
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
        out.push_str("\"format_version\":2,");
        out.push_str("\"dimension\":");
        out.push_str(&self.dimension.to_string());
        out.push_str(",\"scalar\":");
        json::quote(&mut out, S::NAME);
        out.push_str(",\"memory\":");
        push_float(&mut out, self.memory.to_f64());
        out.push_str(",\"eta\":");
        push_float(&mut out, self.eta.to_f64());
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
            out.push_str("]}");
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
        let mut network = Self::new(state.dimension, state.memory, state.eta, budget, universe)?;
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
            layers.push(Layer {
                sigma,
                cells,
                cell_decay: clock,
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
            if layer.cells.iter().any(|cell| zero_scalar_vec(&cell.center)) {
                return Err(Error::Invalid(
                    "persistent CELL center must be nonzero".into(),
                ));
            }
        }
        if self.layers.len() > 1 {
            let n = self.layers.len();
            if self.layers[n - 1].cells.is_empty()
                && self.layers[n - 1].sigma.is_empty()
                && self.layers[n - 2].cells.is_empty()
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
        let universe = universe.into();
        match scalar {
            "f32" => Ok(Self::F32(Auxein::new(
                dimension, memory, eta, budget, universe,
            )?)),
            "f64" => Ok(Self::F64(Auxein::new(
                dimension, memory, eta, budget, universe,
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
}

#[derive(Debug)]
struct ParsedState {
    dimension: usize,
    scalar: String,
    memory: f64,
    eta: f64,
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
        let steps_seen = take_u64(&mut obj, "steps_seen", "state.steps_seen")?;
        let raw_layers = take_array(&mut obj, "layers", "state.layers")?;
        if raw_layers.is_empty() {
            return Err(Error::Invalid("state.layers must be nonempty".into()));
        }
        let mut layers = Vec::with_capacity(raw_layers.len());
        for (li, raw) in raw_layers.into_iter().enumerate() {
            let mut layer = expect_object(raw, &format!("state.layers[{li}]"))?;
            exact_keys(&layer, &["sigma", "cells"], &format!("state.layers[{li}]"))?;
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
            layers.push(ParsedLayer { sigma, cells });
        }
        Ok(Self {
            dimension,
            scalar,
            memory,
            eta,
            steps_seen,
            layers,
        })
    }
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
    if kernels.len() <= 1 || !d0.is_finite() {
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

fn concern_scalar<S: Scalar>(
    kernel: &Kernel<S>,
    x: &[f64],
    input_variance: f64,
    d0_center: f64,
) -> (bool, f64, f64) {
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

fn merge_context_point<S: Scalar>(context: &mut Option<Kernel64>, center: &[S], weight: f64) {
    let point = scalar_vec_to_f64(center);
    match context {
        None => {
            *context = Some(Kernel64 {
                weight,
                center: point,
                variance: 0.0,
            });
        }
        Some(kernel) => {
            let total = kernel.weight + weight;
            let ratio = weight / total;
            let delta2 = stable_sum(kernel.center.iter().zip(&point).map(|(&a, &b)| {
                let d = b - a;
                d * d
            }));
            for (old, &new) in kernel.center.iter_mut().zip(&point) {
                *old = structural_zero(*old + ratio * (new - *old));
            }
            kernel.variance = (kernel.weight * kernel.variance) / total
                + (kernel.weight * weight / (total * total)) * delta2;
            kernel.weight = total;
        }
    }
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

pub fn step_report_json(report: &StepReport) -> String {
    let mut out = String::with_capacity(512);
    out.push('{');
    out.push_str("\"step_index\":");
    out.push_str(&report.step_index.to_string());
    out.push_str(",\"readout\":[");
    for (i, rec) in report.readout.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        json::quote(&mut out, &rec.universe);
        out.push(',');
        write_f64_vec(&mut out, rec.local_input.as_ref());
        out.push(',');
        write_f64_vec(&mut out, &rec.recognised);
        out.push(']');
    }
    out.push_str("],\"transformations\":[");
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
    out.push_str("]}");
    out
}

pub fn summary_json(summary: &Summary) -> String {
    let mut out = String::with_capacity(512);
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
    out.push_str(",\"maintenance_units\":");
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
        Transformation::Promote { layer, count } => {
            out.push_str("{\"phase\":\"geometry\",\"type\":\"promote\",\"layer\":");
            out.push_str(&layer.to_string());
            out.push_str(",\"count\":");
            out.push_str(&count.to_string());
            out.push('}');
        }
        Transformation::GrowthCommit {
            seeds,
            layer_created,
            units,
        } => {
            out.push_str("{\"phase\":\"growth\",\"type\":\"commit\",\"seeds\":");
            out.push_str(&seeds.to_string());
            out.push_str(",\"layer_created\":");
            out.push_str(if *layer_created { "true" } else { "false" });
            out.push_str(",\"units\":");
            out.push_str(&units.to_string());
            out.push('}');
        }
        Transformation::GrowthReject {
            seeds,
            layer_requested,
            units,
        } => {
            out.push_str("{\"phase\":\"growth\",\"type\":\"reject\",\"seeds\":");
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

    #[test]
    fn packing() {
        let n = Auxein::<f64>::new(1, 10.0, 1.0, Budget::kernels("0"), "auxein").unwrap();
        assert_eq!(n.kernel_units().unwrap(), 24);
        assert_eq!(n.network_units().unwrap(), 49);
        assert_eq!(n.min_units().unwrap(), 65);
        assert_eq!(n.budget_units(), 65);
        assert_eq!(n.maintenance_units().unwrap(), 65);
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
    fn f32_state_roundtrip() {
        let n = Auxein::<f32>::new(1, 10.1, 0.7, Budget::kernels("100"), "x").unwrap();
        assert_ne!(n.memory(), 10.1);
        let state = n.export_json();
        let restored =
            Auxein::<f32>::from_json(&state, Budget::units(n.budget_units()), "x").unwrap();
        assert_eq!(restored.export_json(), state);
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
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
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
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":3.0,"C":[2.0],"V":10.0}]}]}"#;
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
        let bad = r#"{"format_version":2,"dimension":1,"scalar":"f32","memory":10.1,"eta":0.7,"steps_seen":0,"layers":[{"sigma":[],"cells":[]}]}"#;
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

            let mut general_targets = Targets::default();
            general_targets.reset(original.len(), x.len(), true);
            for &(index, weight) in &responsibilities {
                general_targets.add_atom(index, &x, 0.0, weight);
            }
            let mut general = original.clone();
            general_targets
                .apply_population(&mut general, beta, lambda)
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
        let state_a = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let state_b = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":100.0,"C":[2.0],"V":10.0}]}]}"#;
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
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![1.0], vec![3.0], vec![-10.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.input_mass, 1.0);
        assert_eq!(layer.output_mass, 2.0 / 3.0);
        assert_eq!(layer.context_center, Some(vec![2.0]));
        assert_eq!(layer.context_variance, Some(1.0));

        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":10.0},{"W":1.0,"C":[2.0],"V":10.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(state, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![3.0]], true).unwrap();
        let layer = &r.layer_reports[0];
        assert_eq!(layer.output_mass, 1.0);
        assert_eq!(layer.context_center, Some(vec![1.5]));
        assert_eq!(layer.context_variance, Some(0.25));
    }

    #[test]
    fn singleton_and_zero_center_contexts_are_vertical_silence() {
        let singleton = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[2.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(singleton, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![2.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![2.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(0.0));
        assert!(!r.layer_reports[0].context_emitted);

        let symmetric = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":0.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-1.0],"V":0.0},{"W":1.0,"C":[1.0],"V":0.0}]}]}"#;
        let mut n = Auxein::<f64>::from_json(symmetric, Budget::kernels("100"), "auxein").unwrap();
        let r = n.step(&[vec![-1.0], vec![1.0]], true).unwrap();
        assert_eq!(r.layer_reports[0].context_center, Some(vec![0.0]));
        assert_eq!(r.layer_reports[0].context_variance, Some(1.0));
        assert!(!r.layer_reports[0].context_emitted);
    }

    #[test]
    fn perfect_pair_emits_one_context_and_stops_after_l1_learns_it() {
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":0.0},{"W":1.0,"C":[3.0],"V":0.0}]}]}"#;
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
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":10.0,"eta":1.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[1.0],"V":1.0},{"W":1.0,"C":[2.0],"V":1.0}]}]}"#;
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
                "{{\"format_version\":2,\"dimension\":1,\"scalar\":\"{scalar}\",\"memory\":23.0,\"eta\":1.0,\"steps_seen\":0,\"layers\":[{{\"sigma\":[],\"cells\":[{{\"W\":1.0,\"C\":[-3.0],\"V\":0.25}},{{\"W\":2.0,\"C\":[1.0],\"V\":0.25}},{{\"W\":3.0,\"C\":[4.0],\"V\":0.25}}]}}]}}"
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
        let state = r#"{"format_version":2,"dimension":1,"scalar":"f64","memory":31.0,"eta":1.0,"steps_seen":0,"layers":[{"sigma":[],"cells":[{"W":1.0,"C":[-3.0],"V":0.25},{"W":2.0,"C":[1.0],"V":0.25},{"W":3.0,"C":[4.0],"V":0.25}]}]}"#;
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
}
