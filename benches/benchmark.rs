use regression_compare::*;
use criterion::{async_executor::FuturesExecutor, *};
use rand::prelude::*;

const NUMBER_OF_DATA_POINTS: usize = 1_000_000;

fn random_data_points(size: usize) -> XY {
    let mut rng = rand::thread_rng();
    let mut xs = Vec::with_capacity(size);
    let mut ys = Vec::with_capacity(size);

    for _ in 1..=size {
        let x: f64 = rng.gen();
        let y: f64 = rng.gen();
        xs.push(x);
        ys.push(y);
    }
    XY::new(xs, ys).unwrap_or_default()
}

fn compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparing linear regression functions");
    let data = random_data_points(NUMBER_OF_DATA_POINTS);
    group.bench_function(
        "async regression",
        |b| {
            b.to_async(FuturesExecutor).iter(|| {
                linear_least_squares(&data)
            });
        },
    );
    group.bench_function("sync regression", |b| b.iter(|| linear_least_squares2(&data)));
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
