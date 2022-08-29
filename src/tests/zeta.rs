use approx::assert_abs_diff_eq;

use super::*;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    wila: Dmat,
    zmat: Tensor3,
    wila_eps: f64,
}

impl Test {
    fn new(dir: &'static str, rows: usize, cols: usize, wila_eps: f64) -> Self {
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
            wila_eps,
        }
    }
}

#[test]
fn test_zeta() {
    let tests = [
        // eps increases with mass, which I guess is from mass dependence of lxm
        // and also wila itself
        Test::new("h2o", 3, 6, 7.6e-7),
        Test::new("h2co", 6, 6, 1.8e-6),
        Test::new("c3h2", 9, 6, 2.8e-6),
        Test::new("c3hf", 9, 6, 4.2e-6),
        Test::new("c3hcn", 12, 6, 6.6e-6),
    ];
    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(&test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2, s.axes);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w: &f64| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, s.n3n, &sqm);
        let (_harms, lxm) = symm_eigen_decomp(fxm);

        let (zmat, wila) = s.zeta(&lxm, &w);

        // println!("{:.8}", wila);
        // println!("{:.8}", test.wila);
        // println!(
        //     "{:.2e}",
        //     (wila.clone().abs() - test.wila.clone().abs()).max()
        // );
        assert_abs_diff_eq!(zmat.abs(), test.zmat.abs(), epsilon = 1e-6);
        assert_abs_diff_eq!(
            wila.abs(),
            test.wila.abs(),
            epsilon = test.wila_eps
        );
    }
}
