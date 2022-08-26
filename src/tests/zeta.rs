use std::fs::read_to_string;

use approx::assert_abs_diff_eq;

use crate::*;

fn load_dmat(filename: &str, rows: usize, cols: usize) -> Dmat {
    let data = read_to_string(filename).unwrap();
    Dmat::from_iterator(
        cols,
        rows,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
    .transpose()
}

#[test]
fn test_zeta() {
    let s = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (_harms, lxm) = symm_eigen_decomp(fxm);
    // TODO test the other return values
    let (zmat, _biga, wila) = s.zeta(s.natom, s.nvib, &lxm, &w);

    // had to swap sign of first row since my LXM has a different sign
    let want_wila = load_dmat("testfiles/h2o/wila", 3, 6);

    // had to negate all of zmat from sign of geometry
    let want_zmat = Tensor3::load("testfiles/h2o/zmat");

    println!("{:.8}", wila);
    println!("{:.8}", want_wila);

    assert_abs_diff_eq!(wila, want_wila, epsilon = 1e-6);
    assert_abs_diff_eq!(zmat, want_zmat, epsilon = 1e-6);
}
