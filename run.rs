use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::f64::consts::PI;

fn run_rcwa_2d(
    lam0: f64,
    theta: f64,
    phi: f64,
    er: Vec<DMatrix<f64>>,
    ur: Vec<DMatrix<f64>>,
    layer_thicknesses: Vec<f64>,
    lattice_constants: (f64, f64),
    pte: f64,
    ptm: f64,
    n: usize,
    m: usize,
    e_half: (f64, f64),
) -> (f64, f64) {
    let normal_vector = DVector::from_vec(vec![0.0, 0.0, -1.0]);
    let ate_vector = DVector::from_vec(vec![0.0, 1.0, 0.0]);

    let (lx, ly) = lattice_constants;
    let nm = (2 * n + 1) * (2 * m + 1);

    let k0 = 2.0 * PI / lam0;

    let mut s_matrices = Vec::new();
    let mut kz_storage = Vec::new();

    let m_r = 1.0;
    let e_r = e_half.0;

    let n_i = (e_r * m_r).sqrt();

    let kx_inc = n_i * theta.sin() * phi.cos();
    let ky_inc = n_i * theta.sin() * phi.sin();
    let kz_inc = Complex::new(e_r, 0.0)
        .sqrt()
        - kx_inc.powi(2)
        - ky_inc.powi(2);

    let (kx, ky) = k_matrix_cubic_2d(kx_inc, ky_inc, k0, lx, ly, n, m); // Implement this function

    let e_h = 1.0;
    let (wg, vg, kzg) = homogeneous_module(kx.clone(), ky.clone(), e_h); // Implement this function

    let (wr, vr, kzr) = homogeneous_module(kx.clone(), ky.clone(), e_r);
    kz_storage.push(kzr.clone());

    let (ar, br) = a_b_matrices(wg.clone(), wr.clone(), vg.clone(), vr.clone()); // Implement this function
    let (s_ref, sr_dict) = s_r(ar, br); // Implement this function
    s_matrices.push(s_ref);
    let mut sg = sr_dict;

    for i in 0..er.len() {
        let e_conv = &er[i];
        let mu_conv = &ur[i];

        let (p, q, kzl) = p_q_kz(kx.clone(), ky.clone(), e_conv.clone(), mu_conv.clone()); // Implement this function
        kz_storage.push(kzl.clone());
        let gamma_squared = &p * &q;

        let (w_i, lambda_matrix) = eigen_w(gamma_squared); // Implement this function
        let v_i = eigen_v(q.clone(), w_i.clone(), lambda_matrix.clone()); // Implement this function

        let (a, b) = a_b_matrices(w_i.clone(), wg.clone(), v_i.clone(), vg.clone());

        let li = layer_thicknesses[i];
        let (s_layer, sl_dict) = s_layer(a, b, li, k0, lambda_matrix); // Implement this function
        s_matrices.push(s_layer);

        sg = redheffer_star(sg, sl_dict); // Implement this function
    }

    let m_t = 1.0;
    let e_t = e_half.1;
    let (wt, vt, kz_trans) = homogeneous_module(kx.clone(), ky.clone(), e_t);

    let (at, bt) = a_b_matrices(wg.clone(), wt.clone(), vg.clone(), vt.clone());
    let (st, st_dict) = s_t(at, bt); // Implement this function
    s_matrices.push(st);
    sg = redheffer_star(sg, st_dict);

    let k_inc_vector = DVector::from_vec(vec![
        n_i * theta.sin() * phi.cos(),
        n_i * theta.sin() * phi.sin(),
        n_i * theta.cos(),
    ]);

    let (e_inc, cinc, polarization) =
        initial_conditions(k_inc_vector, theta, normal_vector, pte, ptm, n, m); // Implement this function

    let cinc = wr.try_inverse().unwrap() * cinc;

    let reflected = &wr * &sg["S11"] * &cinc;
    let transmitted = &wt * &sg["S21"] * &cinc;

    let rx = &reflected.rows(0, nm);
    let ry = &reflected.rows(nm, nm);
    let tx = &transmitted.rows(0, nm);
    let ty = &transmitted.rows(nm, nm);

    let rz = kzr.try_inverse().unwrap() * (kx.clone() * rx + ky.clone() * ry);
    let tz = kz_trans.try_inverse().unwrap() * (kx.clone() * tx + ky.clone() * ty);

    let r_sq = rx.map(|x| x.abs().powi(2))
        + ry.map(|y| y.abs().powi(2))
        + rz.map(|z| z.abs().powi(2));
    let t_sq = tx.map(|x| x.abs().powi(2))
        + ty.map(|y| y.abs().powi(2))
        + tz.map(|z| z.abs().powi(2));

    let r = (kzr.clone().real() * r_sq) / kz_inc.real();
    let t = (kz_trans.real() * t_sq) / kz_inc.real();

    (r.sum(), t.sum())
}

// Helper functions to be implemented:
// - `k_matrix_cubic_2d`
// - `homogeneous_module`
// - `a_b_matrices`
// - `s_r`
// - `p_q_kz`
// - `eigen_w`
// - `eigen_v`
// - `s_layer`
// - `redheffer_star`
// - `s_t`
// - `initial_conditions`
