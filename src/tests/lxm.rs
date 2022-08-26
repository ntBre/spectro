use approx::assert_abs_diff_eq;

use super::*;

use na::dvector;
use nalgebra as na;

#[test]
fn test_lxm() {
    let mut s = Spectro::load("testfiles/h2o/spectro.in");
    s.geom.to_angstrom();
    s.geom.normalize();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let n3n = 3 * s.natoms();
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let want_lxm = load_dmat("testfiles/h2o/lxm", 9, 9);
    let want_harms = dvector![
        3943.69, 3833.70, 1650.93, 30.02, 29.15, 7.33, 0.07, -0.17, -29.61
    ];
    assert_abs_diff_eq!(to_wavenumbers(harms), want_harms, epsilon = 1e-2);
    // check the absolute value since the sign depends on eigen stuff
    assert_abs_diff_eq!(lxm.abs(), want_lxm.abs(), epsilon = 3e-8);
}
