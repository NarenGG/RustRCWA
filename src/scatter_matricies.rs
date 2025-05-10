use nalgebra::{DMatrix, DVector};

fn A(
    w_layer: DMatrix<f64>,
    wg: DMatrix<f64>,
    v_layer: DMatrix<f64>,
    vg: DMatrix<f64>,
) -> DMatrix<f64> {
    w_layer.try_inverse().unwrap() * wg + v_layer.try_inverse().unwrap() * vg
}

fn B(
    w_layer: DMatrix<f64>,
    wg: DMatrix<f64>,
    v_layer: DMatrix<f64>,
    vg: DMatrix<f64>,
) -> DMatrix<f64> {
    w_layer.try_inverse().unwrap() * wg - v_layer.try_inverse().unwrap() * vg
}

fn A_B_matrices_half_space(
    w_layer: DMatrix<f64>,
    wg: DMatrix<f64>,
    v_layer: DMatrix<f64>,
    vg: DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let a = A(wg.clone(), w_layer.clone(), vg.clone(), v_layer.clone());
    let b = B(wg, w_layer, vg, v_layer);
    (a, b)
}

fn A_B_matrices(
    w_layer: DMatrix<f64>,
    wg: DMatrix<f64>,
    v_layer: DMatrix<f64>,
    vg: DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let a = A(w_layer.clone(), wg.clone(), v_layer.clone(), vg.clone());
    let b = B(w_layer, wg, v_layer, vg);
    (a, b)
}

fn S_layer(
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    li: f64,
    k0: f64,
    modes: DMatrix<f64>,
) -> (DMatrix<f64>, [DMatrix<f64>; 4]) {
    let exp_modes = modes.map(|mode| (-mode * li * k0).exp());
    let x_i = DMatrix::from_diagonal(&exp_modes.diagonal());

    let term1 = a.clone()
        - &x_i * &b * a.clone().try_inverse().unwrap() * &x_i * &b;
    let s11 = term1
        .try_inverse()
        .unwrap()
        * (&x_i * &b * a.clone().try_inverse().unwrap() * &x_i * &a - &b);
    let s12 = term1
        .try_inverse()
        .unwrap()
        * (&x_i * (&a - &b * a.clone().try_inverse().unwrap() * &b));
    let s22 = s11.clone();
    let s21 = s12.clone();

    let s_dict = [s11.clone(), s22.clone(), s12.clone(), s21.clone()];
    let s = DMatrix::from_columns(&[s11, s12, s21, s22]);

    (s, s_dict)
}

fn S_R(
    ar: DMatrix<f64>,
    br: DMatrix<f64>,
) -> (DMatrix<f64>, [DMatrix<f64>; 4]) {
    let s11 = -ar.clone().try_inverse().unwrap() * br.clone();
    let s12 = 2.0 * ar.clone().try_inverse().unwrap();
    let s21 = 0.5 * (ar.clone() - br.clone() * ar.clone().try_inverse().unwrap() * br.clone());
    let s22 = br.clone() * ar.clone().try_inverse().unwrap();

    let s_dict = [s11.clone(), s22.clone(), s12.clone(), s21.clone()];
    let s = DMatrix::from_columns(&[s11, s12, s21, s22]);

    (s, s_dict)
}

fn S_T(
    at: DMatrix<f64>,
    bt: DMatrix<f64>,
) -> (DMatrix<f64>, [DMatrix<f64>; 4]) {
    let s11 = bt.clone() * at.clone().try_inverse().unwrap();
    let s21 = 2.0 * at.clone().try_inverse().unwrap();
    let s12 = 0.5 * (at.clone() - bt.clone() * at.clone().try_inverse().unwrap() * bt.clone());
    let s22 = -at.clone().try_inverse().unwrap() * bt.clone();

    let s_dict = [s11.clone(), s22.clone(), s12.clone(), s21.clone()];
    let s = DMatrix::from_columns(&[s11, s12, s21, s22]);

    (s, s_dict)
}