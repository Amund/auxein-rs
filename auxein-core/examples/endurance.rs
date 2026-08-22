use auxein_core::{Budget, Mode, Network};
use std::env;
use std::fs;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

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

fn rss_kib() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    let line = text.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn print_sample(
    net: &Network,
    scalar: &str,
    mode: Mode,
    step: usize,
    elapsed: f64,
) -> Result<(), String> {
    let summary = net.summary().map_err(|e| e.to_string())?;
    let cells: usize = summary.cells_per_layer.iter().sum();
    let sigma: usize = summary.sigma_per_layer.iter().sum();
    let temporal_cells: usize = summary.temporal_cells_per_layer.iter().sum();
    let temporal_sigma: usize = summary.temporal_sigma_per_layer.iter().sum();
    let rss = rss_kib().map_or_else(|| "null".to_string(), |value| value.to_string());
    println!(
        "{{\"step\":{step},\"seconds\":{elapsed},\"scalar\":\"{scalar}\",\"mode\":\"{}\",\"rss_kib\":{rss},\"scratch_capacity_bytes\":{},\"layers\":{},\"cells\":{cells},\"sigma\":{sigma},\"temporal_cells\":{temporal_cells},\"temporal_sigma\":{temporal_sigma}}}",
        mode.as_str(),
        net.transient_memory_capacity_bytes(),
        summary.layer_count,
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let scalar = args.get(1).map(String::as_str).unwrap_or("f64");
    let mode = Mode::parse(args.get(2).map(String::as_str).unwrap_or("geometry"))
        .map_err(|e| e.to_string())?;
    let steps = parse_or(args.get(3), 2_000_000usize, "steps")?;
    let sample_every = parse_or(args.get(4), 250_000usize, "sample interval")?;
    if steps == 0 || sample_every == 0 {
        return Err("steps and sample interval must be positive".into());
    }

    let mut net = Network::new_with_mode(scalar, 1, 20.0, 1.0, mode, Budget::kernels("256"))
        .map_err(|e| e.to_string())?;
    if mode == Mode::Predictive {
        net.begin_sequence(false).map_err(|e| e.to_string())?;
    }

    let start = Instant::now();
    print_sample(&net, scalar, mode, 0, 0.0)?;
    for step in 1..=steps {
        let x = if mode == Mode::Predictive && step % 2 == 0 {
            10.0
        } else {
            1.0
        };
        let presentation = [vec![x]];
        if mode == Mode::Predictive {
            black_box(
                net.sequence_step(&presentation, false)
                    .map_err(|e| e.to_string())?,
            );
        } else {
            black_box(net.step(&presentation, false).map_err(|e| e.to_string())?);
        }
        if step % sample_every == 0 || step == steps {
            print_sample(&net, scalar, mode, step, start.elapsed().as_secs_f64())?;
        }
    }
    if mode == Mode::Predictive {
        net.end_sequence().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("endurance error: {err}");
            ExitCode::from(2)
        }
    }
}
