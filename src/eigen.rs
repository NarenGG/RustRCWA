/*
Functions that analyze the eigenmodes of a medium.
*/

use nalgebra::{DMatrix, ComplexField};
use std::ops::Mul;

pub fn eigen_w(gamma_squared: DMatrix<f64>) -> (DMatrix<ComplexField>, DMatrix<ComplexField>) {
    /*
    For the E_field
    Use: You would only really want to use this if the media is anisotropic in any way.
    :param gamma_squared: Matrix for the scattering formalism.
    :return: Tuple of eigenmodes (W) and the square root of eigenvalues (lambda_matrix).
    */
    let eigen = gamma_squared.clone().eigen();
    let lambda = eigen.eigenvalues;
    let w = eigen.eigenvectors;

    let mut lambda_squared_matrix = DMatrix::zeros(lambda.len(), lambda.len());
    for (i, &value) in lambda.iter().enumerate() {
        lambda_squared_matrix[(i, i)] = value;
    }

    let lambda_matrix = lambda_squared_matrix.map(|x| ComplexField::sqrt(ComplexField::from_real(x)));
    (w, lambda_matrix)
}

pub fn eigen_v(
    q: DMatrix<f64>,
    w: DMatrix<ComplexField>,
    lambda_matrix: DMatrix<ComplexField>,
) -> DMatrix<ComplexField> {
    /*
    Eigenmodes for the i*eta*H field.
    :param q: Q matrix.
    :param w: Modes from eigen_w.
    :param lambda_matrix: Eigenvalues matrix from eigen_w.
    :return: Eigenmodes matrix V.
    */
    q.mul(w).mul(lambda_matrix.try_inverse().expect("Matrix is not invertible"))
}