#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use auxein_core::{
    parse_presentation_json, step_report_json, summary_json, Budget, Error, Mode, Network, Result,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("auxein: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<String> = args.collect();
    match command.as_str() {
        "run" => run_stream(&options),
        "summary" => run_summary(&options),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("auxein {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err(Error::Invalid(format!("unknown command '{command}'"))),
    }
}

#[derive(Default)]
struct Opts {
    dimension: Option<usize>,
    memory: Option<f64>,
    eta: Option<f64>,
    scalar: Option<String>,
    mode: Option<Mode>,
    universe: Option<String>,
    budget: Option<Budget>,
    load: Option<String>,
    save: Option<String>,
    detailed: bool,
}

fn parse_opts(args: &[String], allow_save: bool) -> Result<Opts> {
    let mut out = Opts::default();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        match key.as_str() {
            "--detailed" => {
                out.detailed = true;
                i += 1;
                continue;
            }
            "--dimension" | "--memory" | "--eta" | "--scalar" | "--mode" | "--universe"
            | "--budget" | "--budget-units" | "--load" | "--save" => {}
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(Error::Invalid(format!("unknown option '{key}'"))),
        }
        if key == "--save" && !allow_save {
            return Err(Error::Invalid(
                "--save is not valid for this command".into(),
            ));
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| Error::Invalid(format!("missing value after {key}")))?;
        match key.as_str() {
            "--dimension" => {
                out.dimension = Some(value.parse().map_err(|_| {
                    Error::Invalid("--dimension must be a positive integer".into())
                })?);
            }
            "--memory" => {
                out.memory = Some(
                    value
                        .parse()
                        .map_err(|_| Error::Invalid("--memory must be a real number".into()))?,
                );
            }
            "--eta" => {
                out.eta = Some(
                    value
                        .parse()
                        .map_err(|_| Error::Invalid("--eta must be a real number".into()))?,
                );
            }
            "--scalar" => out.scalar = Some(value.clone()),
            "--mode" => out.mode = Some(Mode::parse(value)?),
            "--universe" => out.universe = Some(value.clone()),
            "--budget" => set_budget_once(&mut out.budget, Budget::kernels(value))?,
            "--budget-units" => {
                let units: u64 = value.parse().map_err(|_| {
                    Error::Invalid("--budget-units must be a nonnegative integer".into())
                })?;
                set_budget_once(&mut out.budget, Budget::units(units))?;
            }
            "--load" => out.load = Some(value.clone()),
            "--save" => out.save = Some(value.clone()),
            _ => unreachable!(),
        }
        i += 2;
    }
    Ok(out)
}

fn set_budget_once(slot: &mut Option<Budget>, value: Budget) -> Result<()> {
    if slot.is_some() {
        return Err(Error::Invalid(
            "provide exactly one of --budget or --budget-units".into(),
        ));
    }
    *slot = Some(value);
    Ok(())
}

fn build_network(opts: &Opts) -> Result<Network> {
    let budget = opts.budget.clone().ok_or_else(|| {
        Error::Invalid("provide exactly one of --budget or --budget-units".into())
    })?;
    let universe = opts.universe.clone().unwrap_or_else(|| "auxein".into());
    if let Some(path) = &opts.load {
        if opts.dimension.is_some()
            || opts.memory.is_some()
            || opts.scalar.is_some()
            || opts.mode.is_some()
        {
            return Err(Error::Invalid(
                "--dimension, --memory, --scalar and --mode come from the loaded state".into(),
            ));
        }
        let text =
            fs::read_to_string(path).map_err(|e| Error::Io(format!("cannot read {path}: {e}")))?;
        let mut network = Network::from_json(&text, budget, universe)?;
        if let Some(eta) = opts.eta {
            network.set_eta(eta)?;
        }
        Ok(network)
    } else {
        let dimension = opts
            .dimension
            .ok_or_else(|| Error::Invalid("--dimension is required for a new state".into()))?;
        let memory = opts
            .memory
            .ok_or_else(|| Error::Invalid("--memory is required for a new state".into()))?;
        let scalar = opts.scalar.as_deref().unwrap_or("f64");
        let eta = opts.eta.unwrap_or(1.0);
        let mode = opts.mode.unwrap_or(Mode::Geometry);
        Network::new_with_mode(scalar, dimension, memory, eta, mode, budget, universe)
    }
}

fn run_stream(args: &[String]) -> Result<()> {
    let opts = parse_opts(args, true)?;
    let mut network = build_network(&opts)?;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for (line_no, line) in stdin.lock().lines().enumerate() {
        let line = line.map_err(|e| Error::Io(format!("stdin: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let presentation = parse_presentation_json(&line)
            .map_err(|e| Error::Invalid(format!("stdin line {}: {e}", line_no + 1)))?;
        let report = network.step(&presentation, opts.detailed)?;
        writeln!(stdout, "{}", step_report_json(&report))
            .map_err(|e| Error::Io(format!("stdout: {e}")))?;
    }
    stdout
        .flush()
        .map_err(|e| Error::Io(format!("stdout: {e}")))?;
    if let Some(path) = &opts.save {
        write_atomic(path, &network.export_json())?;
    }
    Ok(())
}

fn run_summary(args: &[String]) -> Result<()> {
    let opts = parse_opts(args, false)?;
    if opts.load.is_none() {
        return Err(Error::Invalid("summary requires --load FILE".into()));
    }
    let network = build_network(&opts)?;
    println!("{}", summary_json(&network.summary()?));
    Ok(())
}

fn write_atomic(path: &str, contents: &str) -> Result<()> {
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, contents).map_err(|e| Error::Io(format!("cannot write {tmp}: {e}")))?;
    fs::rename(&tmp, path).map_err(|e| Error::Io(format!("cannot replace {path}: {e}")))?;
    Ok(())
}

fn print_help() {
    println!(
        "auxein {version}\n\n\
         Usage:\n\
           auxein run --dimension D --memory T (--budget B | --budget-units N) [options]\n\
           auxein run --load STATE (--budget B | --budget-units N) [options]\n\
           auxein summary --load STATE (--budget B | --budget-units N) [options]\n\n\
         run reads one JSON presentation per stdin line and writes one StepReport JSON per line.\n\n\
         Options:\n\
           --dimension D       vector dimension for a new state\n\
           --memory T          EMA half-life for a new state\n\
           --eta R             learning multiplier in [0,1] (default 1)\n\
           --scalar f32|f64    persistent scalar (default f64)\n\
           --mode geometry|temporal\n\
                              engine mode for a new state (default geometry)\n\
           --budget B          exact-decimal ergonomic kernel capacity\n\
           --budget-units N    exact raw material budget\n\
           --universe NAME     external readout universe (default auxein)\n\
           --load FILE         load canonical format_version=3 JSON state\n\
           --save FILE         atomically save final canonical JSON state\n\
           --detailed          include LayerReport diagnostics\n",
        version = env!("CARGO_PKG_VERSION")
    );
}
