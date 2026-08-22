use auxein_core::{Auxein, Budget, InputAtom, Mode};
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

fn kernel(w: f64, c: &[f64], v: f64) -> String {
    let cs = c
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"W\":{w},\"C\":[{cs}],\"V\":{v}}}")
}

fn layer_json(mode: Mode, sigma: String, cells: String, temporal_cells: String) -> String {
    match mode {
        Mode::Geometry => format!("{{\"sigma\":[{sigma}],\"cells\":[{cells}]}}"),
        Mode::Predictive => format!(
            "{{\"sigma\":[{sigma}],\"cells\":[{cells}],\"temporal_sigma\":[],\"temporal_cells\":[{temporal_cells}],\"previous\":null}}"
        ),
    }
}

fn axis(d: usize, value: f64) -> Vec<f64> {
    let mut c = vec![0.0; d];
    c[0] = value;
    c
}

fn push_kernel_list(out: &mut String, value: String) {
    if !out.is_empty() {
        out.push(',');
    }
    out.push_str(&value);
}

fn require_predictive(scenario: &str, mode: Mode) -> Result<(), String> {
    if mode == Mode::Predictive {
        Ok(())
    } else {
        Err(format!("{scenario} requires predictive mode"))
    }
}

fn state_json(
    scenario: &str,
    d: usize,
    cells: usize,
    mode: Mode,
    eta: f64,
) -> Result<String, String> {
    if d == 0 {
        return Err("dimension must be positive".into());
    }
    if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
        return Err("eta must lie in [0,1]".into());
    }

    let layers = match scenario {
        "singleton" | "weighted-partial" => {
            let c = axis(d, 2.0);
            layer_json(mode, String::new(), kernel(1.0, &c, 0.25), String::new())
        }
        "predictive-stable" => {
            require_predictive(scenario, mode)?;
            let c = axis(d, 2.0);
            let target = axis(d, 3.0);
            let mut tc = c.clone();
            tc.extend_from_slice(&target);
            layer_json(
                mode,
                String::new(),
                kernel(1.0, &c, 0.25),
                kernel(1.0, &tc, 0.5),
            )
        }
        "predictive-sequence" => {
            require_predictive(scenario, mode)?;
            let a = axis(d, 1.0);
            let b = axis(d, 10.0);
            layer_json(
                mode,
                String::new(),
                format!("{},{}", kernel(1.0, &a, 0.0), kernel(1.0, &b, 0.0)),
                String::new(),
            )
        }
        "predictive-fanout" | "predictive-duplicate-fanout" | "temporal-outside" => {
            require_predictive(scenario, mode)?;
            let source = axis(d, 2.0);
            let mut temporal = String::new();
            for i in 0..cells {
                let temporal_source = if scenario == "temporal-outside" {
                    axis(d, 1000.0 + i as f64)
                } else if scenario == "predictive-duplicate-fanout" {
                    // Distinct temporal CELL geometries with one shared future.
                    // Tiny source offsets keep every relation locally relevant
                    // while respecting canonical no-clone state packing.
                    axis(d, 2.0 + i as f64 * 1e-6)
                } else {
                    source.clone()
                };
                let target = if scenario == "predictive-duplicate-fanout" {
                    axis(d, 3.0)
                } else {
                    axis(d, 10.0 + i as f64)
                };
                let mut center = temporal_source;
                center.extend_from_slice(&target);
                push_kernel_list(&mut temporal, kernel(1.0, &center, 0.25));
            }
            layer_json(mode, String::new(), kernel(1.0, &source, 0.25), temporal)
        }
        "pair-context" => {
            let a = axis(d, 1.0);
            let b = axis(d, 3.0);
            let c = axis(d, 2.0);
            format!(
                "{},{}",
                layer_json(
                    mode,
                    String::new(),
                    format!("{},{}", kernel(1.0, &a, 0.0), kernel(1.0, &b, 0.0)),
                    String::new(),
                ),
                layer_json(mode, String::new(), kernel(1.0, &c, 1.0), String::new(),)
            )
        }
        "sparse" | "dense" | "same-first" => {
            let mut ks = String::new();
            for i in 0..cells {
                let c = match scenario {
                    "sparse" => {
                        let mut c = axis(d, 10.0 + i as f64 * 0.01);
                        for (j, x) in c.iter_mut().enumerate().skip(1) {
                            *x = ((i * 17 + j * 13) % 97) as f64 * 1e-6;
                        }
                        c
                    }
                    "dense" => {
                        let angle = std::f64::consts::TAU * i as f64 / cells.max(1) as f64;
                        let mut c = axis(d, 1.0 + 0.01 * angle.cos());
                        if d > 1 {
                            c[1] = 0.01 * angle.sin();
                        }
                        c
                    }
                    "same-first" => {
                        if d < 2 {
                            return Err("same-first requires dimension >= 2".into());
                        }
                        let mut c = axis(d, 1.0);
                        c[d - 1] = 100.0 + i as f64;
                        c
                    }
                    _ => return Err("internal benchmark scenario mismatch".into()),
                };
                let variance = if scenario == "dense" { 10.0 } else { 0.01 };
                push_kernel_list(&mut ks, kernel(1.0, &c, variance));
            }
            layer_json(mode, String::new(), ks, String::new())
        }
        "sigma-idle" => {
            let mut sigma = String::new();
            for i in 0..cells {
                push_kernel_list(&mut sigma, kernel(0.5, &axis(d, 1000.0 + i as f64), 0.0));
            }
            layer_json(mode, sigma, kernel(1.0, &axis(d, 2.0), 1.0), String::new())
        }
        _ => return Err(format!("unknown scenario '{scenario}'")),
    };

    Ok(format!(
        "{{\"format_version\":5,\"dimension\":{d},\"scalar\":\"f64\",\"memory\":50.0,\"eta\":{eta},\"mode\":\"{}\",\"steps_seen\":0,\"layers\":[{layers}]}}",
        mode.as_str()
    ))
}

enum Presentation {
    Uniform(Vec<Vec<f64>>),
    Weighted(Vec<InputAtom>),
}

fn presentation(
    scenario: &str,
    d: usize,
    cells: usize,
    index: usize,
) -> Result<Presentation, String> {
    let p = match scenario {
        "singleton"
        | "predictive-stable"
        | "predictive-fanout"
        | "predictive-duplicate-fanout"
        | "temporal-outside"
        | "sigma-idle" => Presentation::Uniform(vec![axis(d, 2.0)]),
        "weighted-partial" => Presentation::Weighted(vec![
            InputAtom::new(0.25, axis(d, 2.0), 0.0),
            InputAtom::new(0.75, axis(d, 20.0), 0.0),
        ]),
        "predictive-sequence" => {
            Presentation::Uniform(vec![axis(d, if index % 2 == 0 { 1.0 } else { 10.0 })])
        }
        "pair-context" => Presentation::Uniform(vec![axis(d, 1.0), axis(d, 3.0)]),
        "sparse" => Presentation::Uniform(vec![axis(d, 10.0 + (cells / 2) as f64 * 0.01)]),
        "dense" => Presentation::Uniform(vec![axis(d, 2.0)]),
        "same-first" => {
            if d < 2 {
                return Err("same-first requires dimension >= 2".into());
            }
            Presentation::Uniform(vec![axis(d, 1.0)])
        }
        _ => return Err(format!("unknown scenario '{scenario}'")),
    };
    Ok(p)
}

fn atomic_step(n: &mut Auxein<f64>, p: &Presentation) -> Result<(), String> {
    match p {
        Presentation::Uniform(v) => n.step(v, false).map(black_box).map_err(|e| e.to_string()),
        Presentation::Weighted(v) => n
            .step_weighted(v, false)
            .map(black_box)
            .map_err(|e| e.to_string()),
    }
    .map(|_| ())
}

fn sequence_step(n: &mut Auxein<f64>, p: &Presentation) -> Result<(), String> {
    match p {
        Presentation::Uniform(v) => n
            .sequence_step(v, false)
            .map(black_box)
            .map_err(|e| e.to_string()),
        Presentation::Weighted(v) => n
            .sequence_step_weighted(v, false)
            .map(black_box)
            .map_err(|e| e.to_string()),
    }
    .map(|_| ())
}

fn parse_or<T: std::str::FromStr>(
    value: Option<&String>,
    default: T,
    label: &str,
) -> Result<T, String> {
    match value {
        Some(value) => value
            .parse()
            .map_err(|_| format!("invalid {label}: {value}")),
        None => Ok(default),
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = env::args().collect();
    let scenario = a.get(1).map(String::as_str).unwrap_or("singleton");
    let d = parse_or(a.get(2), 8usize, "dimension")?;
    let cells = parse_or(a.get(3), 512usize, "cells")?;
    let steps = parse_or(a.get(4), 100_000usize, "steps")?;
    let warmup = parse_or(a.get(5), 1_000usize, "warmup")?;
    if steps == 0 {
        return Err("steps must be positive".into());
    }
    let mode = Mode::parse(a.get(6).map(String::as_str).unwrap_or("geometry"))
        .map_err(|e| e.to_string())?;
    let eta = parse_or(a.get(7), 0.0f64, "eta")?;
    let causal = scenario == "predictive-sequence";
    let end = warmup
        .checked_add(steps)
        .ok_or_else(|| "warmup + steps overflow".to_string())?;

    let state = state_json(scenario, d, cells, mode, eta)?;
    let budget_kernels = (cells as u128)
        .checked_mul(3)
        .and_then(|v| v.checked_add(1000))
        .ok_or_else(|| "budget sizing overflow".to_string())?;
    let mut n = Auxein::<f64>::from_json(&state, Budget::kernels(budget_kernels.to_string()))
        .map_err(|e| e.to_string())?;

    if causal {
        n.begin_sequence(false).map_err(|e| e.to_string())?;
        for i in 0..warmup {
            sequence_step(&mut n, &presentation(scenario, d, cells, i)?)?;
        }
        let t = Instant::now();
        for i in warmup..end {
            sequence_step(&mut n, &presentation(scenario, d, cells, i)?)?;
        }
        let secs = t.elapsed().as_secs_f64();
        n.end_sequence().map_err(|e| e.to_string())?;
        println!(
            "{{\"canon\":\"0.5.0\",\"mode\":\"{}\",\"scenario\":\"{scenario}\",\"dimension\":{d},\"cells\":{cells},\"eta\":{eta},\"steps\":{steps},\"seconds\":{secs},\"microseconds_per_presentation\":{},\"presentations_per_second\":{}}}",
            mode.as_str(),
            secs * 1e6 / steps as f64,
            steps as f64 / secs
        );
        return Ok(());
    }

    for i in 0..warmup {
        atomic_step(&mut n, &presentation(scenario, d, cells, i)?)?;
    }
    let t = Instant::now();
    for i in warmup..end {
        atomic_step(&mut n, &presentation(scenario, d, cells, i)?)?;
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{{\"canon\":\"0.5.0\",\"mode\":\"{}\",\"scenario\":\"{scenario}\",\"dimension\":{d},\"cells\":{cells},\"eta\":{eta},\"steps\":{steps},\"seconds\":{secs},\"microseconds_per_presentation\":{},\"presentations_per_second\":{}}}",
        mode.as_str(),
        secs * 1e6 / steps as f64,
        steps as f64 / secs
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("benchmark error: {err}");
            ExitCode::from(2)
        }
    }
}
