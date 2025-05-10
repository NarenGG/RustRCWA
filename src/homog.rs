extern crate nalgebra as na;

use na::{DMatrix, Complex};
use std::f64::consts::PI;

fn homogeneous_module(
    kx: DMatrix<f64>,
    ky: DMatrix<f64>,
    e_r: f64,
    m_r: f64,
) -> (DMatrix<f64>, DMatrix<Complex<f64>>, DMatrix<Complex<f64>>) {
    assert!(kx.is_square(), "kx must be a square matrix");
    assert!(ky.is_square(), "ky must be a square matrix");

    let j = Complex::new(0.0, 1.0); // Imaginary unit i
    let n = kx.nrows(); // Number of rows in kx
    let identity = DMatrix::<f64>::identity(n, n);

    // Matrix P
    let p = (1.0 / e_r)
        * DMatrix::from_fn(2 * n, 2 * n, |i, j| {
            if i < n && j < n {
                kx[(i, j)] * ky[(i, j)]
            } else if i < n && j >= n {
                e_r * m_r * identity[(i, j - n)] - kx[(i, j - n)] * kx[(i, j - n)]
            } else if i >= n && j < n {
                ky[(i - n, j)] * ky[(i - n, j)] - m_r * e_r * identity[(i - n, j)]
            } else {
                -ky[(i - n, j - n)] * kx[(i - n, j - n)]
            }
        });

    // Matrix Q
    let q = (e_r / m_r) * &p;

    // Identity matrix W
    let w = DMatrix::<f64>::identity(2 * n, 2 * n);

    // Argument for Kz
    let mut arg = m_r * e_r * &identity - &kx * &kx - &ky * &ky;
    let arg_complex = arg.map(|x| Complex::new(x, 0.0));

    // Kz computation
    let kz = arg_complex.map(|x| x.sqrt().conj()); // Conjugate for negative sign convention

    // Eigenvalues
    let eigenvalues = DMatrix::from_fn(2 * n, 2 * n, |i, j| {
        if i == j {
            if i < n {
                j * kz[(i, i)]
            } else {
                j * kz[(i - n, i - n)]
            }
        } else {
            Complex::new(0.0, 0.0)
        }
    });

    // Matrix V
    let v = &q * eigenvalues.try_inverse().unwrap(); // Alternative if inverse fails

    (w, v, kz)
}

fn homogeneous_1d(
    kx: DMatrix<f64>,
    e_r: f64,
    m_r: f64,
) -> (DMatrix<f64>, DMatrix<Complex<f64>>, DMatrix<Complex<f64>>) {
    let j = Complex::new(0.0, 1.0); // Imaginary unit i
    let identity = DMatrix::<f64>::identity(kx.nrows(), kx.ncols());

    // Matrix P
    let p = e_r * &identity - &kx * &kx;

    // Matrix Q
    let q = identity.clone();

    // Argument for Kz
    let mut arg = m_r * e_r * &identity - &kx * &kx;
    let arg_complex = arg.map(|x| Complex::new(x, 0.0));

    // Kz computation
    let kz = arg_complex.map(|x| x.sqrt().conj()); // Conjugate for negative sign convention

    // Eigenvalues
    let eigenvalues = kz.map(|x| j * x);

    // Matrix V
    let v = &q * eigenvalues;

    (identity, v, kz)
}