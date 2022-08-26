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

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    wila: Dmat,
    zmat: Tensor3,
}

impl Test {
    fn new(dir: &'static str, rows: usize, cols: usize) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            wila: load_dmat(
                start.join(dir).join("wila").to_str().unwrap(),
                rows,
                cols,
            ),
            zmat: Tensor3::load(start.join(dir).join("zmat").to_str().unwrap()),
        }
    }
}

#[test]
fn test_zeta() {
    let tests = [
        // had to swap sign of first row of wila since my LXM has a different
        // sign, and had to negate all of zmat from sign of geometry
        Test::new("h2o", 3, 6),
    ];
    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(&test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2, s.axes);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, s.n3n, &sqm);
        let (_harms, lxm) = symm_eigen_decomp(fxm);

        let (zmat, wila) = s.zeta(&lxm, &w);

        assert_abs_diff_eq!(wila, test.wila, epsilon = 1e-6);
        assert_abs_diff_eq!(zmat, test.zmat, epsilon = 1e-6);
    }
}
