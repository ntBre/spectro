use approx::assert_abs_diff_eq;

use crate::*;

use super::load_dmat;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    fort30: String,
    fort40: String,
    xcnst: Dmat,
    e0: f64,
}

impl Test {
    fn new(dir: &'static str, nvib: usize, e0: f64) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            fort30: String::from(
                start.join(dir).join("fort.30").to_str().unwrap(),
            ),
            fort40: String::from(
                start.join(dir).join("fort.40").to_str().unwrap(),
            ),
            xcnst: load_dmat(start.join(dir).join("xcnst"), nvib, nvib),
            e0,
        }
    }
}

#[test]
fn test_xcalc() {
    let tests = [
        Test::new("h2o", 3, 20.057563725859055),
        Test::new("h2co", 6, 11.49172492996696),
        Test::new("c3h2", 9, 4.2142433303609623),
        Test::new("c3hf", 9, -2.5183180568351426),
        Test::new("c3hcn", 12, -5.6448536013927253),
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = s.make_lx(s.n3n, &sqm, &lxm);
        let (zmat, _) = s.zeta(&lxm, &w);
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x, s.axes);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
        let f4x = load_fc4(test.fort40, s.n3n);
        let mut f4x = s.rot4th(f4x, s.axes);
        let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq);
        let Restst {
            coriolis: _,
            fermi1,
            fermi2,
            darling: _,
            i1sts: _,
            i1mode: _,
        } = s.restst(&zmat, &f3qcm, &freq);
        let (xcnst, e0) = xcalc(
            s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &s.rotcon, &fermi1, &fermi2,
        );
        // println!("\n{}", test.infile);
        // println!("got\n{:.12}", xcnst);
        // println!("want\n{:.12}", test.xcnst);
        // println!(
        //     "xcnst diff = {:.2e}",
        //     (xcnst.clone() - test.xcnst.clone()).abs().max()
        // );
        // println!("e0 diff = {:.2e}", (e0 - test.e0).abs());
        assert_abs_diff_eq!(xcnst, test.xcnst, epsilon = 1.54e-5);
        // NOTE 6e-8 works for everything but c3hcn, might want to investigate
        assert_abs_diff_eq!(e0, test.e0, epsilon = 1.4e-7);
    }
}
