use std::{fs::read_to_string, path::Path};

use approx::assert_abs_diff_eq;

use crate::{
    quartic::Quartic,
    utils::{load_fc2, symm_eigen_decomp, to_wavenumbers},
    Spectro, FACT2,
};

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    want: Quartic,
}

impl Test {
    fn new(dir: &'static str) -> Self {
        let start = Path::new("testfiles");
        let data =
            read_to_string(start.join(dir).join("quartic.json")).unwrap();
        let want: Quartic = serde_json::from_str(&data).unwrap();
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            want,
        }
    }
}

#[test]
fn test_quartic() {
    let tests = [
        Test::new("h2o"),
        Test::new("h2co"),
        Test::new("c3h2"),
        Test::new("c3hf"),
        Test::new("c3hcn"),
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w: &f64| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(&harms);
        let (_zmat, wila) = s.zeta(&lxm, &w);
        let got = Quartic::new(&s, &freq, &wila);
        // println!("got\n{}", got);
        // println!("want\n{}", test.want);
        // println!("diff\n{}", got.clone() - test.want.clone());

        // accept this size of epsilon because this is about how good the
        // rotational constant agreement is and b[xyz][as] are the largest
        // differences
        assert_abs_diff_eq!(got, test.want, epsilon = 2e-5);
    }
}
