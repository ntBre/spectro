use approx::assert_abs_diff_eq;
use nalgebra::dmatrix;

use crate::FACT2;

use super::*;

#[test]
fn test_find3r() {
    let got = find3r(2, 2, 2);
    let want = 9;
    assert_eq!(got, want);
}

#[test]
fn test_quartic_sum_facs() {
    let got = quartic_sum_facs(15, 3);
    let want: Vec<f64> = vec![
        24.0, 6.0, 4.0, 6.0, 24.0, 6.0, 2.0, 2.0, 6.0, 4.0, 2.0, 4.0, 6.0, 6.0,
        24.0,
    ];
    assert_eq!(got, want);
}

#[test]
fn test_taupcm() {
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let (_zmat, _biga, wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
    let primat = s.geom.principal_moments();
    let tau = make_tau(3, 3, &freq, &primat, &wila);
    let got = tau_prime(3, &tau);
    let want = dmatrix![
    -0.08628870,  0.01018052, -0.00283749;
     0.01018052, -0.00839612, -0.00138895;
    -0.00283749, -0.00138895, -0.00093412;
       ];
    assert_abs_diff_eq!(got, want, epsilon = 2e-7);
}
