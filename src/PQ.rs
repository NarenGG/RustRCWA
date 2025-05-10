use ndarray::{Array2, Array, ArrayView2};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::error::LinalgError;

// Function to calculate Q Matrix
fn q_matrix(
    kx: &Array2<f64>,
    ky: &Array2<f64>,
    e_conv: &Array2<f64>,
    mu_conv: &Array2<f64>,
) -> Result<Array2<f64>, LinalgError> {
    // Assuming that mu_conv is invertible
    let inv_mu_conv = mu_conv.clone().inv()?;

    let upper_left = kx.dot(&inv_mu_conv.dot(ky));
    let upper_right = e_conv - kx.dot(&inv_mu_conv.dot(kx));
    let lower_left = ky.dot(&inv_mu_conv.dot(ky)) - e_conv;
    let lower_right = -ky.dot(&inv_mu_conv.dot(kx));

    let q = ndarray::stack(
        ndarray::Axis(0),
        &[
            ndarray::stack(
                ndarray::Axis(1),
                &[upper_left.view(), upper_right.view()],
            )?,
            ndarray::stack(
                ndarray::Axis(1),
                &[lower_left.view(), lower_right.view()],
            )?,
        ],
    )?;

    Ok(q)
}

// Function to calculate P Matrix
fn p_matrix(
    kx: &Array2<f64>,
    ky: &Array2<f64>,
    e_conv: &Array2<f64>,
    mu_conv: &Array2<f64>,
) -> Result<Array2<f64>, LinalgError> {
    // Assuming that e_conv is invertible
    let inv_e_conv = e_conv.clone().inv()?;

    let upper_left = kx.dot(&inv_e_conv.dot(ky));
    let upper_right = mu_conv - kx.dot(&inv_e_conv.dot(kx));
    let lower_left = ky.dot(&inv_e_conv.dot(ky)) - mu_conv;
    let lower_right = -ky.dot(&inv_e_conv.dot(kx));

    let p = ndarray::stack(
        ndarray::Axis(0),
        &[
            ndarray::stack(
                ndarray::Axis(1),
                &[upper_left.view(), upper_right.view()],
            )?,
            ndarray::stack(
                ndarray::Axis(1),
                &[lower_left.view(), lower_right.view()],
            )?,
        ],
    )?;

    Ok(p)
}

// Function to calculate P, Q, and Kz
fn p_q_kz(
    kx: &Array2<f64>,
    ky: &Array2<f64>,
    e_conv: &Array2<f64>,
    mu_conv: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<Complex<f64>>), LinalgError> {
    let argument = e_conv - &(kx.mapv(|x| x.powi(2))) - &(ky.mapv(|x| x.powi(2)));
    let kz = argument.mapv(|x| Complex::new(x, 0.0).sqrt());

    let q = q_matrix(kx, ky, e_conv, mu_conv)?;
    let p = p_matrix(kx, ky, e_conv, mu_conv)?;

    Ok((p, q, kz))
}