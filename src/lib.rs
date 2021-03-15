use futures::*;
use std::{iter::Zip, slice::Iter};

#[derive(Default)]
pub struct XY {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl XY {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Option<Self> {
        let xlen = xs.len();
        if xlen == ys.len() && xlen >= 3 {
            Some(Self { xs, ys })
        } else {
            None
        }
    }

    pub fn get(&self, n: usize) -> Option<(f64, f64)> {
        self.xs.get(n).zip(self.ys.get(n)).map(|(x, y)| (*x, *y))
    }

    pub fn keys(&self) -> &[f64] {
        &self.xs
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn values(&self) -> &[f64] {
        &self.ys
    }

    pub fn iter(&self) -> Zip<Iter<'_, f64>, Iter<'_, f64>> {
        self.keys().iter().zip(self.values().iter())
    }
}

type DataPoints = XY;

pub async fn sum(values: &[f64]) -> f64 {
    values.iter().sum::<f64>()
}

pub fn square(n: f64) -> f64 {
    n * n
}

pub async fn sum_x(data_points: &DataPoints) -> f64 {
    data_points.keys().iter().sum::<f64>()
}

pub async fn sum_x_squares(data_points: &DataPoints) -> f64 {
    data_points.keys().iter().map(|x| x * x).sum::<f64>()
}

pub async fn sum_xy(data_points: &DataPoints) -> f64 {
    data_points.iter().map(|(x, y)| x * y).sum::<f64>()
}

pub async fn sum_y(data_points: &DataPoints) -> f64 {
    data_points.values().iter().sum::<f64>()
}

pub async fn ss_tot(data_points: &DataPoints, y_mean: f64) -> f64 {
    data_points.values().iter().map(|y| square(y - y_mean)).sum::<f64>()
}

pub async fn ss_res(data_points: &DataPoints, fun: impl Fn(f64) -> f64) -> f64 {
    data_points.iter().map(|(x, y)| square(y - fun(*x))).sum::<f64>()
}

pub struct LinearModel {
    slope: f64,
    intersect: f64,
    r_square: f64
}

impl LinearModel {
    pub fn new(slope: f64, intersect: f64, r_square: f64) -> Self {
        Self { slope, intersect, r_square }
    }

    pub fn slope(&self) -> f64 {
        self.slope
    }

    pub fn intersect(&self) -> f64 {
        self.intersect
    }

    pub fn as_function(&self) -> impl Fn(f64) -> f64 {
        let slope = self.slope;
        let intersect = self.intersect;
        move |x| slope * x + intersect
    }

    pub fn r_square(&self) -> f64 {
        self.r_square
    }
}

#[allow(clippy::suspicious_operation_groupings)]
pub async fn linear_least_squares(data_points: &DataPoints) -> LinearModel {
    let len = data_points.len() as f64;
    let (sum_x, sum_x_squares, sum_xy, sum_y) = join![
        sum_x(data_points),
        sum_x_squares(data_points),
        sum_xy(data_points),
        sum_y(data_points)
    ];
    let slope = (len * sum_xy - sum_x * sum_y) / (len * sum_x_squares - sum_x * sum_x);
    let intersect = (sum_y - slope * sum_x) / len;
    let fun = |x: f64| slope * x + intersect;
    let (ss_tot, ss_res) = join![ss_tot(data_points, sum_y / len), ss_res(data_points, fun)];
    let r_square = 1.0 - ss_res / ss_tot;

    LinearModel::new(slope, intersect, r_square)
}

#[allow(clippy::suspicious_operation_groupings)]
pub fn linear_least_squares2(data_points: &DataPoints) -> LinearModel {
    let len = data_points.len() as f64;
    let sum_x = data_points.keys().iter().sum::<f64>();
    let sum_x_squares = data_points.keys().iter().map(|x| x * x).sum::<f64>();
    let sum_xy = data_points.iter().map(|(x, y)| x * y).sum::<f64>();
    let sum_y = data_points.values().iter().sum::<f64>();
        
    let slope = (len * sum_xy - sum_x * sum_y) / (len * sum_x_squares - sum_x * sum_x);
    let intersect = (sum_y - slope * sum_x) / len;
    let fun = |x: f64| slope * x + intersect;
    let y_mean = sum_y / len;
    let ss_tot = data_points.values().iter().map(|y| square(y - y_mean)).sum::<f64>();
    let ss_res = data_points.iter().map(|(x, y)| square(y - fun(*x))).sum::<f64>();
    let r_square = 1.0 - ss_res / ss_tot;

    LinearModel::new(slope, intersect, r_square)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use futures::executor::block_on;

    #[test]
    fn lest_squares_test() {
        let data = DataPoints::new(vec![1.0, 2.0, 3.0], vec![3.0, 6.0, 9.0]).unwrap();

        let model = block_on(linear_least_squares(&data));
        assert_eq!(3.0, model.slope());
        assert_eq!(0.0, model.intersect());
        assert_eq!(1.0, model.r_square());
        let model_as_function = model.as_function();

        assert_eq!(12.0, model_as_function(4.0));

        let data2 = DataPoints::new(vec![1.0, 2.0, 3.0], vec![4.0, 7.0, 10.0]).unwrap();

        let model2 = block_on(linear_least_squares(&data2));
        assert_eq!(3.0, model2.slope());
        assert_eq!(1.0, model2.intersect());
        assert_eq!(1.0, model2.r_square());

        let model_as_function2 = model2.as_function();

        assert_eq!(7.0, model_as_function2(2.0));
    }

    #[test]
    fn lest_squares2_test() {
        let data = DataPoints::new(vec![1.0, 2.0, 3.0], vec![3.0, 6.0, 9.0]).unwrap();

        let model = linear_least_squares2(&data);
        assert_eq!(3.0, model.slope());
        assert_eq!(0.0, model.intersect());
        assert_eq!(1.0, model.r_square());
        let model_as_function = model.as_function();

        assert_eq!(12.0, model_as_function(4.0));

        let data2 = DataPoints::new(vec![1.0, 2.0, 3.0], vec![4.0, 7.0, 10.0]).unwrap();

        let model2 = linear_least_squares2(&data2);
        assert_eq!(3.0, model2.slope());
        assert_eq!(1.0, model2.intersect());
        assert_eq!(1.0, model2.r_square());

        let model_as_function2 = model2.as_function();

        assert_eq!(7.0, model_as_function2(2.0));
    }
}
