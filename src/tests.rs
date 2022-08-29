use std::fs::read_to_string;

use approx::assert_abs_diff_eq;

use crate::*;

use na::dmatrix;
use nalgebra as na;

mod force3;
mod force4;
mod load;
mod lxm;
mod quartic;
mod restst;
mod run;
mod zeta;

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
fn test_load_fc2() {
    let got = load_fc2("testfiles/fort.15", 9);
    let want = dmatrix![
    3.2416100000000003e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.37452424350000002, 0.23501356370000001, 0.0,
    -0.34333094920000001,
    -0.20370165770000001, 0.0, -0.031192384199999999, -0.031313796200000001;
    0.0, 0.23501356370000001, 0.21818269339999999, -2.7999999999999998e-09,
    -0.26632741869999998, -0.2298216998, 0.0, 0.031313796200000001,
    0.011639992;
    0.0, 0.0, -2.7999999999999998e-09, 6.36953e-05, 0.0,
    2.7999999999999998e-09, -3.0777900000000002e-05, 0.0, 0.0;
    0.0, -0.34333094920000001, -0.26632741869999998, 0.0,
    0.68666005870000002,
    0.0, 0.0, -0.34333094920000001, 0.26632741869999998;
    0.0, -0.20370165770000001, -0.2298216998, 2.7999999999999998e-09, 0.0,
    0.45964140840000001, 0.0, 0.20370165770000001, -0.2298216998;
    0.0, 0.0, 0.0, -3.0777900000000002e-05, 0.0, 0.0,
    3.2416100000000003e-05,
    0.0, 2.7999999999999998e-09;
    0.0, -0.031192384199999999, 0.031313796200000001, 0.0,
    -0.34333094920000001, 0.20370165770000001, 0.0, 0.37452424350000002,
    -0.23501356649999999;
    0.0, -0.031313796200000001, 0.011639992, 0.0, 0.26632741869999998,
    -0.2298216998, 2.7999999999999998e-09, -0.23501356649999999,
    0.21818269339999999;
        ];
    assert_eq!(got, want);
}

#[test]
fn test_rot2nd() {
    let spectro = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let got = spectro.rot2nd(fc2);
    let want = dmatrix![
    0.37452424350000002, 0.23501356370000001, 0.0, -0.34333094920000001,
    -0.20370165770000001, 0.0, -0.031192384199999999, -0.031313796200000001, 0.0;
    0.23501356370000001, 0.21818269339999999, 0.0, -0.26632741869999998,
    -0.2298216998, -2.7999999999999998e-09, 0.031313796200000001, 0.011639992, 0.0;
     0.0, 0.0, 3.2416100000000003e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    -0.34333094920000001, -0.26632741869999998, 0.0, 0.68666005870000002,
    0.0, 0.0, -0.34333094920000001, 0.26632741869999998, 0.0;
    -0.20370165770000001, -0.2298216998, 0.0, 0.0, 0.45964140840000001,
    2.7999999999999998e-09, 0.20370165770000001, -0.2298216998, 0.0;
    0.0, -2.7999999999999998e-09, 0.0, 0.0, 2.7999999999999998e-09, 6.36953e-05,
    0.0, 0.0, -3.0777900000000002e-05;
    -0.031192384199999999, 0.031313796200000001, 0.0, -0.34333094920000001,
    0.20370165770000001, 0.0, 0.37452424350000002, -0.23501356649999999, 0.0;
    -0.031313796200000001, 0.011639992, 0.0, 0.26632741869999998,
    -0.2298216998, 0.0, -0.23501356649999999, 0.21818269339999999,
    2.7999999999999998e-09;
     0.0, 0.0, 0.0, 0.0, 0.0, -3.0777900000000002e-05, 0.0,
     2.7999999999999998e-09, 3.2416100000000003e-05;
           ];
    assert_eq!(got, want);
}

#[test]
fn test_sec() {
    let spectro = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = spectro.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let got = spectro.form_sec(fc2, &sqm);
    let mut want = dmatrix![
    5.7857638, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    3.6305606, 3.3705523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0000000, 0.0000000, 0.0005008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    -1.3313595, -1.0327573, 0.0000000, 0.6683836, 0.0, 0.0, 0.0, 0.0, 0.0;
    -0.7899088, -0.8911964, 0.0000000, 0.0000000, 0.4474074, 0.0, 0.0, 0.0, 0.0;
    0.0000000, -0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000620, 0.0, 0.0, 0.0;
    -0.4818694, 0.4837450, 0.0000000, -1.3313595, 0.7899088, 0.0000000, 5.7857638, 0.0, 0.0;
    -0.4837450, 0.1798181, 0.0000000, 1.0327573, -0.8911964, 0.0000000, -3.6305607, 3.3705523, 0.0;
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, -0.0001193, 0.0000000, 0.0000000, 0.0005008;
       ];
    want.fill_upper_triangle_with_lower_triangle();
    assert_abs_diff_eq!(got, want, epsilon = 1e-7);
}

#[test]
fn test_funds_and_e0() {
    #[derive(Clone)]
    struct Test {
        infile: &'static str,
        fort15: &'static str,
        fort30: &'static str,
        fort40: &'static str,
        want_e0: f64,
        want_fund: Vec<f64>,
        e_eps: f64,
    }
    let tests = [
        Test {
            infile: "testfiles/h2o/spectro.in",
            fort15: "testfiles/h2o/fort.15",
            fort30: "testfiles/h2o/fort.30",
            fort40: "testfiles/h2o/fort.40",
            want_e0: 20.057563725859055,
            want_fund: vec![3753.166, 3656.537, 1598.516],
            e_eps: 6e-8,
        },
        Test {
            infile: "testfiles/h2co/spectro.in",
            fort15: "testfiles/h2co/fort.15",
            fort30: "testfiles/h2co/fort.30",
            fort40: "testfiles/h2co/fort.40",
            want_e0: 11.49172492996696,
            want_fund: vec![
                2842.9498684325331,
                2780.0927479723691,
                1747.8239712488792,
                1499.4165366400482,
                1246.8067957023538,
                1166.9312314784524,
            ],
            e_eps: 6e-8,
        },
        Test {
            infile: "testfiles/c3h2/spectro.in",
            fort15: "testfiles/c3h2/fort.15",
            fort30: "testfiles/c3h2/fort.30",
            fort40: "testfiles/c3h2/fort.40",
            want_e0: 4.2142433303609623,
            want_fund: vec![
                3140.1433372410634,
                3113.2905071971295,
                1589.145000387535,
                1273.1343017671454,
                1059.5183326828769,
                967.49835061508804,
                887.12367115864799,
                846.15735546423639,
                769.62107643057936,
            ],
            e_eps: 1e-8,
        },
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = s.make_lx(s.n3n, &sqm, &lxm);
        let (zmat, _wila) = s.zeta(&lxm, &w);
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x, s.axes);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
        let f4x = load_fc4(test.fort40, s.n3n);
        let mut f4x = s.rot4th(f4x, s.axes);
        let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
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
        assert_abs_diff_eq!(e0, test.want_e0, epsilon = test.e_eps);

        let got = funds(&freq, s.nvib, &xcnst);
        assert_abs_diff_eq!(
            Dvec::from(got),
            Dvec::from(test.want_fund),
            epsilon = 1e-3
        );
    }
}

#[test]
fn test_enrgy() {
    let s = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, _wila) = s.zeta(&lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let f4x = load_fc4("testfiles/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
    let (xcnst, e0) =
        xcalc(s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &s.rotcon, &[], &[]);
    let wante0 = 20.057563725859055;
    assert_abs_diff_eq!(e0, wante0, epsilon = 6e-8);
    let fund = funds(&freq, s.nvib, &xcnst);
    let Restst {
        coriolis: _,
        fermi1,
        fermi2,
        darling: _,
        i1sts,
        i1mode,
    } = s.restst(&zmat, &f3qcm, &freq);
    let mut got = vec![0.0; i1sts.len()];
    enrgy(
        &fund, &freq, &xcnst, &f3qcm, e0, &i1sts, &i1mode, &fermi1, &fermi2,
        &mut got,
    );
    // my numbers after comparing visually to fortran
    let want = vec![
        4656.438188555293,
        8409.60462543482,
        8312.975664427879,
        6254.953686350812,
        12065.411263606182,
        11883.428277109273,
        7818.958032793332,
        11899.804775421886,
        9988.129295095654,
        9895.66935813587,
    ];
    assert_eq!(got, want);
}

#[test]
fn test_alpha() {
    let s = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, wila) = s.zeta(&lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let Restst {
        coriolis,
        fermi1: _,
        fermi2: _,
        darling: _,
        i1sts: _,
        i1mode: _,
    } = s.restst(&zmat, &f3qcm, &freq);
    let got =
        s.alpha(&s.rotcon, &freq, &wila, &s.primat, &zmat, &f3qcm, &coriolis);
    let want = dmatrix![
       -1.1564648277177876, -0.68900933871743675, 2.5983329447479688;
    -0.09868782762986765, -0.21931495793096034, 0.16185995232026804;
    -0.14297483172390801, -0.17703802340977196, -0.14639237714841669;
                                                                         ];
    assert_abs_diff_eq!(got, want.transpose(), epsilon = 3e-6);
}

#[test]
fn test_alphaa() {
    let s = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, wila) = s.zeta(&lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let f4x = load_fc4("testfiles/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
    let (xcnst, _e0) =
        xcalc(s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &s.rotcon, &[], &[]);
    let fund = funds(&freq, s.nvib, &xcnst);
    let Restst {
        coriolis,
        fermi1: _,
        fermi2: _,
        darling: _,
        i1sts,
        i1mode,
    } = s.restst(&zmat, &f3qcm, &freq);
    let got = s.alphaa(
        &s.rotcon, &freq, &wila, &zmat, &f3qcm, &fund, &i1mode, &i1sts,
        &coriolis,
    );
    let want = dmatrix![
    27.657417987118755, 14.498766626639174, 9.2673038449583238;
     26.500953159400968, 14.400078799009306, 9.1243290132344157;
     26.968408648401319, 14.279451668708212, 9.0902658215485506;
     30.255750931866725, 14.660626578959441, 9.1209114678099059;
    ];
    assert_abs_diff_eq!(got, want, epsilon = 2e-5);
}
