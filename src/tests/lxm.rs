use approx::assert_abs_diff_eq;

use super::*;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    lxm: Dmat,
    harm: Vec<f64>,
}

impl Test {
    fn new(
        dir: &'static str,
        rows: usize,
        cols: usize,
        harm: Vec<f64>,
    ) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            lxm: load_dmat(
                start.join(dir).join("lxm").to_str().unwrap(),
                rows,
                cols,
            ),
            harm,
        }
    }
}

#[test]
fn test_lxm() {
    let tests = [Test::new(
        "h2o",
        9,
        9,
        vec![
            3943.6903070625431,
            3833.7018985135023,
            1650.9329629443762,
            30.024414884101041,
            29.153600077627246,
            7.3277291693141766,
            0.072620302917321009,
            // these two are imaginary so negate
            -0.17306628504479143,
            -29.613852000401547,
        ],
    )];
    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(&test.fort15, 9);
        let fc2 = s.rot2nd(fc2, s.axes);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, s.n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        assert_abs_diff_eq!(
            to_wavenumbers(harms),
            Dvec::from(test.harm),
            epsilon = 6e-6
        );
        // check the absolute value since the sign depends on eigen stuff
        assert_abs_diff_eq!(lxm.abs(), test.lxm.abs(), epsilon = 3e-8);
    }
}
