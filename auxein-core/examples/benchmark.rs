use auxein_core::{Auxein, Budget};
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn state_json(scenario: &str, d: usize, cells: usize) -> String {
    let layers = match scenario {
        "singleton" => {
            let mut c = vec![0.0; d];
            c[0] = 2.0;
            format!("{{\"sigma\":[],\"cells\":[{}]}}", kernel(1.0, &c, 0.25))
        }
        "pair-context" => {
            let mut a = vec![0.0; d];
            a[0] = 1.0;
            let mut b = vec![0.0; d];
            b[0] = 3.0;
            let mut c = vec![0.0; d];
            c[0] = 2.0;
            format!(
                "{{\"sigma\":[],\"cells\":[{},{}]}},{{\"sigma\":[],\"cells\":[{}]}}",
                kernel(1.0, &a, 0.0),
                kernel(1.0, &b, 0.0),
                kernel(1.0, &c, 1.0)
            )
        }
        "sparse" => {
            let mut ks = String::new();
            for i in 0..cells {
                if i > 0 {
                    ks.push(',');
                }
                let mut c = vec![0.0; d];
                c[0] = 10.0 + i as f64 * 0.01;
                for (j, x) in c.iter_mut().enumerate().skip(1) {
                    *x = ((i * 17 + j * 13) % 97) as f64 * 1e-6;
                }
                ks.push_str(&kernel(1.0, &c, 0.01));
            }
            format!("{{\"sigma\":[],\"cells\":[{ks}]}}")
        }
        "dense" => {
            let mut ks = String::new();
            for i in 0..cells {
                if i > 0 {
                    ks.push(',');
                }
                let angle = std::f64::consts::TAU * i as f64 / cells.max(1) as f64;
                let mut c = vec![0.0; d];
                c[0] = 1.0 + 0.01 * angle.cos();
                if d > 1 {
                    c[1] = 0.01 * angle.sin();
                }
                ks.push_str(&kernel(1.0, &c, 10.0));
            }
            format!("{{\"sigma\":[],\"cells\":[{ks}]}}")
        }
        _ => panic!("unknown scenario"),
    };
    format!("{{\"format_version\":2,\"dimension\":{d},\"scalar\":\"f64\",\"memory\":50.0,\"eta\":0.0,\"steps_seen\":0,\"layers\":[{layers}]}}")
}

fn kernel(w: f64, c: &[f64], v: f64) -> String {
    let cs = c
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"W\":{w},\"C\":[{cs}],\"V\":{v}}}")
}
fn presentation(scenario: &str, d: usize, cells: usize) -> Vec<Vec<f64>> {
    match scenario {
        "singleton" => {
            let mut x = vec![0.0; d];
            x[0] = 2.0;
            vec![x]
        }
        "pair-context" => {
            let mut a = vec![0.0; d];
            a[0] = 1.0;
            let mut b = vec![0.0; d];
            b[0] = 3.0;
            vec![a, b]
        }
        "sparse" => {
            let mut x = vec![0.0; d];
            x[0] = 10.0 + (cells / 2) as f64 * 0.01;
            vec![x]
        }
        "dense" => {
            let mut x = vec![0.0; d];
            x[0] = 2.0;
            vec![x]
        }
        _ => unreachable!(),
    }
}
fn main() {
    let a: Vec<String> = env::args().collect();
    let scenario = a.get(1).map(String::as_str).unwrap_or("singleton");
    let d = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(8usize);
    let cells = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(512usize);
    let steps = a
        .get(4)
        .and_then(|x| x.parse().ok())
        .unwrap_or(100_000usize);
    let warmup = a.get(5).and_then(|x| x.parse().ok()).unwrap_or(1_000usize);
    let state = state_json(scenario, d, cells);
    let mut n = Auxein::<f64>::from_json(
        &state,
        Budget::kernels((cells * 3 + 1000).to_string()),
        "auxein",
    )
    .unwrap();
    let p = presentation(scenario, d, cells);
    for _ in 0..warmup {
        black_box(n.step(&p, false).unwrap());
    }
    let t = Instant::now();
    for _ in 0..steps {
        black_box(n.step(&p, false).unwrap());
    }
    let secs = t.elapsed().as_secs_f64();
    println!("{{\"scenario\":\"{scenario}\",\"dimension\":{d},\"cells\":{cells},\"steps\":{steps},\"seconds\":{secs},\"microseconds_per_step\":{},\"steps_per_second\":{}}}",secs*1e6/steps as f64,steps as f64/secs);
}
