use ndarray::{Array1, Array2, s};
use ndarray_linalg::Diag;

fn k_matrix_cubic_2d(
    beta_x: f64,
    beta_y: f64,
    k0: f64,
    a_x: f64,
    a_y: f64,
    n_p: i32,
    n_q: i32,
) -> (Array2<f64>, Array2<f64>) {
    // Generate the k_x and k_y arrays
    let range_p: Vec<f64> = (-n_p..=n_p).map(|p| p as f64).collect();
    let range_q: Vec<f64> = (-n_q..=n_q).map(|q| q as f64).collect();

    let k_x: Array1<f64> = Array1::from(range_p)
        .mapv(|p| beta_x - 2.0 * std::f64::consts::PI * p / (k0 * a_x));
    let k_y: Array1<f64> = Array1::from(range_q)
        .mapv(|q| beta_y - 2.0 * std::f64::consts::PI * q / (k0 * a_y));

    // Create the kx and ky grids
    let kx = k_x.broadcast((range_q.len(), range_p.len())).unwrap();
    let ky = k_y.broadcast((range_p.len(), range_q.len())).unwrap();

    // Flatten the grids and create diagonal matrices
    let kx_flat = kx.iter().cloned().collect::<Array1<f64>>();
    let ky_flat = ky.iter().cloned().collect::<Array1<f64>>();

    let kx_diag = Array2::from_diag(&kx_flat);
    let ky_diag = Array2::from_diag(&ky_flat);

    (kx_diag, ky_diag)
}