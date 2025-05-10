use ndarray::{Array, Array2, Array1};
use ndarray::linalg::Dot;
use std::f64;

// Create a delta matrix for 2D
fn delta_vector(p: usize, q: usize) -> Array2<f64> {
    let mut fourier_grid = Array2::<f64>::zeros((p, q));
    fourier_grid[[p / 2, q / 2]] = 1.0;
    return fourier_grid;
}

// Create a delta vector for 1D
fn delta_vector_1d(p: usize) -> Array1<f64> {
    let mut vector = Array1::<f64>::zeros(p);
    let index = p / 2;
    vector[index] = 1.0;
    return vector;
}

// 1D Initial Conditions
fn initial_conditions_1d(k_inc_vector: &Array1<f64>, theta: f64, p: usize) -> Array1<f64> {
    let num_ord = 2 * p + 1;
    let delta = delta_vector_1d(num_ord);
    let cinc = delta.clone();
    cinc
}

// 2D Initial Conditions
fn initial_conditions(
    k_inc_vector: &Array1<f64>,
    theta: f64,
    normal_vector: &Array1<f64>,
    pte: f64,
    ptm: f64,
    p: usize,
    q: usize,
) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    let ate_vector = if theta != 0.0 {
        let mut cross_product = k_inc_vector.cross(normal_vector);
        cross_product /= cross_product.norm();
        cross_product
    } else {
        Array1::from(vec![0.0, 1.0, 0.0])
    };

    let mut atm_vector = ate_vector.cross(k_inc_vector);
    atm_vector /= atm_vector.norm();

    let polarization = pte * &ate_vector + ptm * &atm_vector;
    let e_inc = polarization.clone();
    let polarization = polarization.to_owned(); // Squeeze as scalar/vector representation
    let delta = delta_vector(2 * p + 1, 2 * q + 1);

    let esrc = Array2::hstack(&[
        delta.dot(polarization[0].c2),
        polarization ).[polaruzation[0 )].
}
``