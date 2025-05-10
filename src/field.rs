use ndarray::{Array2, Array};
use num_complex::Complex;
use std::f64::consts::PI;

fn get_field_ref(
    rx: &Array2<Complex<f64>>,
    ry: &Array2<Complex<f64>>,
    rz: &Array2<Complex<f64>>,
    kx: &Array2<f64>,
    ky: &Array2<f64>,
    kz: &Array2<f64>,
    k0: f64,
    field_size: f64,
    field_pts: usize,
    dz: f64,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array2<Complex<f64>>) {
    let nm = kx.shape()[0];
    let mut ex = Array2::<Complex<f64>>::zeros((field_pts, field_pts));
    let mut ey = Array2::<Complex<f64>>::zeros((field_pts, field_pts));
    let mut ez = Array2::<Complex<f64>>::zeros((field_pts, field_pts));

    let xxx: Vec<f64> = (0..field_pts)
        .map(|i| i as f64 * field_size / (field_pts as f64 - 1.0))
        .collect();
    let yyy = xxx.clone();

    for (i, &x) in xxx.iter().enumerate() {
        for (j, &y) in yyy.iter().enumerate() {
            let mut ex_val = Complex::new(0.0, 0.0);
            let mut ey_val = Complex::new(0.0, 0.0);
            let mut ez_val = Complex::new(0.0, 0.0);
            for n in 0..nm {
                let kxmn = kx[[n, n]] * k0;
                let kymn = ky[[n, n]] * k0;
                let kzmn = kz[[n, n]] * k0;

                let phase = Complex::new(0.0, -1.0) * Complex::new(kxmn * x + kymn * y - kzmn * dz, 0.0);
                ex_val += rx[[n, 0]] * phase.exp();
                ey_val += ry[[n, 0]] * phase.exp();
                ez_val += rz[[n, 0]] * phase.exp();
            }
            ex[[i, j]] = ex_val;
            ey[[i, j]] = ey_val;
            ez[[i, j]] = ez_val;
        }
    }
    (ex, ey, ez)
}

// Implement `get_field_trans` and `get_field_phc` similarly to `get_field_ref`