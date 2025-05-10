extern crate nalgebra as na;
use na::{DMatrix, Dynamic, Matrix, Matrix4, Identity};
use std::collections::HashMap;

/// Converts a dictionary (HashMap) of matrices to a block matrix.
fn dict_to_matrix(s_dict: HashMap<&str, DMatrix<f64>>) -> DMatrix<f64> {
    let s11 = s_dict.get("S11").expect("Key 'S11' not found");
    let s12 = s_dict.get("S12").expect("Key 'S12' not found");
    let s21 = s_dict.get("S21").expect("Key 'S21' not found");
    let s22 = s_dict.get("S22").expect("Key 'S22' not found");

    DMatrix::from_fn(s11.nrows() + s21.nrows(), s11.ncols() + s12.ncols(), |i, j| {
        if i < s11.nrows() && j < s11.ncols() {
            s11[(i, j)]
        } else if i < s11.nrows() {
            s12[(i, j - s11.ncols())]
        } else if j < s11.ncols() {
            s21[(i - s11.nrows(), j)]
        } else {
            s22[(i - s11.nrows(), j - s11.ncols())]
        }
    })
}

/// Implements the Redheffer Star Product for two scattering matrices.
fn redheffer_star(sa: HashMap<&str, DMatrix<f64>>, sb: HashMap<&str, DMatrix<f64>>) -> (DMatrix<f64>, HashMap<&'static str, DMatrix<f64>>) {
    let sa_11 = sa.get("S11").expect("Key 'S11' not found in SA");
    let sa_12 = sa.get("S12").expect("Key 'S12' not found in SA");
    let sa_21 = sa.get("S21").expect("Key 'S21' not found in SA");
    let sa_22 = sa.get("S22").expect("Key 'S22' not found in SA");

    let sb_11 = sb.get("S11").expect("Key 'S11' not found in SB");
    let sb_12 = sb.get("S12").expect("Key 'S12' not found in SB");
    let sb_21 = sb.get("S21").expect("Key 'S21' not found in SB");
    let sb_22 = sb.get("S22").expect("Key 'S22' not found in SB");

    let n = sa_11.nrows();
    let identity = DMatrix::<f64>::identity(n, n);

    let d = identity.clone() - sb_11 * sa_22;
    let f = identity.clone() - sa_22 * sb_11;

    let sab_11 = sa_11 + sa_12 * d.clone().lu().solve(&(sb_11 * sa_21)).unwrap();
    let sab_12 = sa_12 * d.clone().lu().solve(sb_12).unwrap();
    let sab_21 = sb_21 * f.clone().lu().solve(sa_21).unwrap();
    let sab_22 = sb_22 + sb_21 * f.clone().lu().solve(&(sa_22 * sb_12)).unwrap();

    let sab = dict_to_matrix(HashMap::from([
        ("S11", sab_11.clone()),
        ("S12", sab_12.clone()),
        ("S21", sab_21.clone()),
        ("S22", sab_22.clone()),
    ]));

    let sab_dict = HashMap::from([
        ("S11", sab_11),
        ("S12", sab_12),
        ("S21", sab_21),
        ("S22", sab_22),
    ]);

    (sab, sab_dict)
}

/// Constructs a global scattering matrix from a list of individual scattering matrices.
fn construct_global_scatter(scatter_list: Vec<HashMap<&str, DMatrix<f64>>>) -> HashMap<&'static str, DMatrix<f64>> {
    let mut sg = scatter_list[0].clone();

    for scatter in scatter_list.iter().skip(1) {
        sg = redheffer_star(sg, scatter.clone()).1;
    }

    sg
}

fn main() {
    // Example usage of the functions can be added here.
}