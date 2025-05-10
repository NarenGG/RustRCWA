use ndarray::{Array2, Array};
use ndarray::prelude::*; 
use rustfft::{FFTplanner, num_complex::Complex};
use std::f64;

// Function to perform FFT shift
fn fftshift(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (nrows, ncols) = input.dim();
    let mut output = Array2::zeros((nrows, ncols));

    let row_mid = nrows / 2;
    let col_mid = ncols / 2;

    for i in 0..nrows {
        for j in 0..ncols {
            let new_i = (i + row_mid) % nrows;
            let new_j = (j + col_mid) % ncols;
            output[[new_i, new_j]] = input[[i, j]];
        }
    }

    output
}

fn convmat2d(a: Array2<f64>, p: usize, q: usize) -> Array2<Complex<f64>> {
    let n = a.dim();
    let nh = (2 * p + 1) * (2 * q + 1);
    let mut c = Array2::<Complex<f64>>::zeros((nh, nh));

    let p_range: Vec<isize> = (-p as isize..=p as isize).collect();
    let q_range: Vec<isize> = (-q as isize..=q as isize).collect();

    // Perform FFT
    let mut af = Array2::<Complex<f64>>::zeros((n.0, n.1));
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(n.1);
    let mut input: Vec<Complex<f64>> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut output = vec![Complex::<f64>::zero(); n.1];
    fft.process(&mut input, &mut output);
    af.assign(&Array2::from_shape_vec((n.0, n.1), output).unwrap());
    af = fftshift(&af) / (n.0 * n.1) as f64;

    // Central indices
    let p0 = (n.1 / 2) as isize;
    let q0 = (n.0 / 2) as isize;

    for (qrow, &qval_row) in q_range.iter().enumerate() {
        for (prow, &pval_row) in p_range.iter().enumerate() {
            let row = qrow * (2 * p + 1) + prow;
            for (qcol, &qval_col) in q_range.iter().enumerate() {
                for (pcol, &pval_col) in p_range.iter().enumerate() {
                    let col = qcol * (2 * p + 1) + pcol;
                    let pfft = pval_row - pval_col;
                    let qfft = qval_row - qval_col;
                    let af_val = af[(q0 + qfft) as usize][(p0 + pfft) as usize];
                    c[[row, col]] = af_val;
                }
            }
        }
    }

    c
}