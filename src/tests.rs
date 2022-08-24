use std::str::FromStr;

use approx::assert_abs_diff_eq;
use symm::Molecule;

use crate::Curvil::*;
use crate::*;

use na::{dmatrix, dvector, matrix, DVector};
use nalgebra as na;

#[test]
fn load() {
    let got = Spectro::load("testfiles/spectro.in");
    let want = Spectro {
        header: vec![
            1, 1, 5, 2, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        geom: Molecule::from_str(
            "
C      0.00000000 -0.89784221  0.00000000
C      0.67631628  0.36939882  0.00000000
C     -0.67631628  0.36939882  0.00000000
H      1.55759109  0.94685817  0.00000000
H     -1.55759109  0.94685817  0.00000000
",
        )
        .unwrap(),
        weights: vec![
            (1, 12.0),
            (2, 12.0),
            (3, 12.0),
            (4, 1.007825),
            (5, 1.007825),
        ],
        curvils: vec![
            Bond(2, 3),
            Bond(1, 2),
            Bond(1, 3),
            Bond(2, 4),
            Bond(3, 5),
            Bend(2, 4, 1),
            Bend(3, 5, 1),
            Tors(4, 2, 1, 3),
            Tors(5, 3, 1, 2),
        ],
        degmodes: vec![],
        dummies: vec![],
        rotor: Rotor::AsymmTop,
        n3n: 15,
        i3n3n: 680,
        i4n3n: 3060,
        nvib: 9,
        i2vib: 45,
        i3vib: 165,
        i4vib: 495,
        natom: 5,
        axes: matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        0.0, 0.0, 1.0;
        ],
    };
    assert_eq!(got, want);
}

#[test]
fn load_dummy() {
    let got = Spectro::load("testfiles/dummy.in");
    let want = Spectro {
        header: vec![
            1, 1, 3, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        geom: Molecule::from_str(
            "
He    -1.76798258  0.00000000  0.00000000
H     -0.53078563  0.00000000  0.00000000
H      0.53078563  0.00000000  0.00000000
He     1.76798258  0.00000000  0.00000000
",
        )
        .unwrap(),
        weights: vec![
            (1, 4.00260325413),
            (2, 1.007825),
            (3, 1.007825),
            (4, 4.00260325413),
            (5, 0.00),
            (6, 0.00),
            (7, 0.00),
            (8, 0.00),
        ],
        curvils: vec![
            Bond(1, 2),
            Bond(2, 3),
            Bond(3, 4),
            Bend(2, 1, 3),
            Bend(3, 2, 4),
        ],
        degmodes: vec![vec![3, 2, 0], vec![1, 2, 3], vec![4, 6], vec![5, 7]],
        dummies: vec![
            Dummy {
                x: DummyVal::Value(1.1111111111),
                y: DummyVal::Atom(3),
                z: DummyVal::Atom(1),
            },
            Dummy {
                x: DummyVal::Atom(3),
                y: DummyVal::Value(1.1111111111),
                z: DummyVal::Atom(1),
            },
            Dummy {
                x: DummyVal::Value(1.1111111111),
                y: DummyVal::Atom(3),
                z: DummyVal::Atom(2),
            },
            Dummy {
                y: DummyVal::Value(1.1111111111),
                x: DummyVal::Atom(3),
                z: DummyVal::Atom(2),
            },
        ],
        rotor: Rotor::Linear,
        n3n: 12,
        i3n3n: 364,
        i4n3n: 1365,
        nvib: 7,
        i2vib: 28,
        i3vib: 84,
        i4vib: 210,
        natom: 4,
        axes: matrix![
        0.0, 0.0, 1.0;
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        ],
    };
    assert_eq!(got, want);
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
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();
    spectro.geom.normalize();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let got = spectro.rot2nd(fc2, spectro.axes);
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
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();
    spectro.geom.normalize();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = spectro.rot2nd(fc2, spectro.axes);
    let fc2 = FACT2 * fc2;
    let n3n = 3 * spectro.natoms();
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let got = spectro.form_sec(fc2, n3n, &sqm);
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
fn test_normfx() {
    let mut s = Spectro::load("testfiles/h2o.in");
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
    // factorization isn't unique so test against mine after visually inspecting
    let want_lxm = dmatrix![
    0.53796936, -0.57484775, 0.41176461, 0.00000844, 0.00000000, 0.00000920,
    -0.23728677, -0.02835492, 0.39176515;
    0.41727449, -0.38803972, -0.54172804, -0.00000780, 0.00000000, 0.00000625,
    0.03156561, -0.23446545, -0.57084803;
    -0.00000000, -0.00000000, 0.00000000, -0.00000000, 1.00000000, 0.00000000,
    -0.00000000, 0.00000000, 0.00000000;
    -0.27007752, 0.00000001, 0.00000000, -0.00000410, 0.00000000, -0.00000227,
    -0.93449808, -0.11167133, -0.20322642;
    0.00000001, 0.19480902, 0.27195827, 0.00001760, -0.00000000, 0.00007510,
    0.11181660, -0.93572709, 0.00000731;
    -0.00000000, 0.00000000, 0.00000000, -0.24654949, -0.00000000, 0.96913020,
    -0.00000642, 0.00007238, -0.00001610;
    0.53796940, 0.57484771, -0.41176460, 0.00000843, 0.00000000, 0.00000920,
    -0.23728746, -0.02834916, 0.39176516;
    -0.41727452, -0.38803968, -0.54172803, 0.00001661, -0.00000000, 0.00003145,
    0.02456907, -0.23529259, 0.57085171;
    -0.00000000, -0.00000000, -0.00000001, 0.96913020, 0.00000000, 0.24654948,
    -0.00000365, 0.00003757, -0.00002616;
                   ];
    let want_harms = dvector![
        3943.69, 3833.70, 1650.93, 30.02, 29.15, 7.33, 0.07, -0.17, -29.61
    ];
    assert_abs_diff_eq!(to_wavenumbers(harms), want_harms, epsilon = 1e-2);
    assert_abs_diff_eq!(lxm, want_lxm, epsilon = 1e-7);
}

#[test]
fn test_run() {
    #[derive(Clone)]
    struct Test {
        infile: &'static str,
        fort15: &'static str,
        fort30: &'static str,
        fort40: &'static str,
        want: Output,
    }
    let tests = [
        Test {
            infile: "testfiles/h2o.in",
            fort15: "testfiles/fort.15",
            fort30: "testfiles/fort.30",
            fort40: "testfiles/fort.40",
            want: Output {
                harms: dvector![3943.7, 3833.7, 1650.9],
                funds: vec![3753.2, 3656.5, 1598.5],
                rots: vec![
                    Rot::new(vec![0, 0, 0], 27.65578, 14.50450, 9.26320),
                    Rot::new(vec![1, 0, 0], 26.49932, 14.40582, 9.12023),
                    Rot::new(vec![0, 1, 0], 26.96677, 14.28519, 9.08617),
                    Rot::new(vec![0, 0, 1], 30.25411, 14.66636, 9.11681),
                ],
                corrs: vec![3753.2, 3656.5, 1598.5],
            },
        },
        Test {
            infile: "testfiles/h2co.in",
            fort15: "testfiles/h2co.15",
            fort30: "testfiles/h2co.30",
            fort40: "testfiles/h2co.40",
            want: Output {
                harms: dvector![3004.6, 2932.6, 1778.7, 1534.1, 1269.8, 1186.9],
                funds: vec![2842.9, 2780.1, 1747.8, 1499.4, 1246.8, 1166.9],
                rots: vec![
                    Rot::new(vec![0, 0, 0, 0, 0, 0], 9.39885, 1.29151, 1.13102),
                    Rot::new(vec![1, 0, 0, 0, 0, 0], 9.30907, 1.29000, 1.12956),
                    Rot::new(vec![0, 1, 0, 0, 0, 0], 9.23556, 1.29111, 1.12877),
                    Rot::new(vec![0, 0, 1, 0, 0, 0], 9.39585, 1.28435, 1.12225),
                    Rot::new(vec![0, 0, 0, 1, 0, 0], 9.46174, 1.30001, 1.12842),
                    Rot::new(vec![0, 0, 0, 0, 1, 0], 9.51923, 1.29405, 1.12472),
                    Rot::new(vec![0, 0, 0, 0, 0, 1], 9.25579, 1.27912, 1.13234),
                ],
                corrs: vec![2826.6, 2778.4, 1747.8, 1499.4, 1246.8, 1166.9],
            },
        },
        Test {
            infile: "testfiles/c3h2.in",
            fort15: "testfiles/c3h2.15",
            fort30: "testfiles/c3h2.30",
            fort40: "testfiles/c3h2.40",
            want: Output {
                harms: dvector![
                    3281.2, 3247.5, 1623.3, 1307.6, 1090.7, 993.0, 908.5,
                    901.5, 785.4
                ],
                funds: vec![
                    3140.1, 3113.3, 1589.1, 1273.1, 1059.5, 967.5, 887.1,
                    846.2, 769.6,
                ],
                rots: vec![
                    Rot::new(
                        vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
                        1.16424,
                        1.06982,
                        0.55583,
                    ),
                    Rot::new(
                        vec![1, 0, 0, 0, 0, 0, 0, 0, 0],
                        1.16232,
                        1.06543,
                        0.55420,
                    ),
                    Rot::new(
                        vec![0, 1, 0, 0, 0, 0, 0, 0, 0],
                        1.16233,
                        1.06571,
                        0.55431,
                    ),
                    Rot::new(
                        vec![0, 0, 1, 0, 0, 0, 0, 0, 0],
                        1.16451,
                        1.06367,
                        0.55439,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 1, 0, 0, 0, 0, 0],
                        1.16114,
                        1.07017,
                        0.55620,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
                        1.15978,
                        1.07109,
                        0.55131,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 0, 0, 1, 0, 0, 0],
                        1.16512,
                        1.06609,
                        0.55637,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 0, 0, 0, 1, 0, 0],
                        1.16260,
                        1.07277,
                        0.55390,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 0, 0, 0, 0, 1, 0],
                        1.16618,
                        1.07216,
                        0.55534,
                    ),
                    Rot::new(
                        vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
                        1.15985,
                        1.06958,
                        0.55663,
                    ),
                ],
                corrs: vec![
                    3128.0, 3113.3, 1590.5, 1273.1, 1059.5, 967.5, 887.1,
                    846.2, 769.6,
                ],
            },
        },
    ];
    for test in Vec::from(&tests[..]) {
        let spectro = Spectro::load(test.infile);
        let got = spectro.run(test.fort15, test.fort30, test.fort40);
        assert_abs_diff_eq!(got.harms, test.want.harms, epsilon = 0.1);
        assert_abs_diff_eq!(
            Dvec::from(got.funds),
            Dvec::from(test.want.funds),
            epsilon = 0.1
        );
        assert_abs_diff_eq!(
            Dvec::from(got.corrs),
            Dvec::from(test.want.corrs),
            epsilon = 0.1
        );
        assert_abs_diff_eq!(
            DVector::from(got.rots),
            DVector::from(test.want.rots),
            epsilon = 3e-5
        );
    }
}

#[test]
fn test_zeta() {
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (_harms, lxm) = symm_eigen_decomp(fxm);
    // TODO test the other return values
    let (_zmat, _biga, got) = s.zeta(s.natom, s.nvib, &lxm, &w);
    // had to swap sign of first row since my LXM has a different sign
    let want = dmatrix![
     -3.6123785074337889e-08,      -1.2692098865561809,  -6.6462962133861936e-08,   -2.9957242484449652e-09,  1.8035956714004908e-09,  -1.0258674709717752e-07 ;
        -0.9155081475600737  ,4.7344317466446739e-08  ,   -1.7484950796657917  ,-2.9480138066186021e-09  ,  2.318919812342448e-09  ,   -2.6640032272258658 ;
        -1.2781035077371032  ,2.1818979178966913e-08  ,    1.2524505892097131  ,-2.2198838796685475e-08  , 1.7461636243495606e-08  ,  -0.02565291852739008 ;
    ];
    assert_abs_diff_eq!(got, want, epsilon = 1e-6);
}

#[test]
fn test_force3() {
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let got = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    // signs are different from fortran version, but I think that's okay. mostly
    // a regression test anyway
    let want = vec![
        0.00028013699326642446,
        1822.24785969789,
        -8.833731114767919e-5,
        1820.1092348470625,
        266.1186010180256,
        -4.814160093191156e-5,
        75.42085160498344,
        -0.0003478413270064172,
        -310.8977456046787,
        -269.0192202516248,
    ];
    assert_eq!(got, want);
}

#[test]
fn test_force4() {
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let f4x = load_fc4("testfiles/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    let got = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
    // signs different from fortran again, probably okay
    let want = vec![
        769.3358855094393,
        -0.002712912620238425,
        763.1685129051438,
        -0.001501171784219294,
        758.2529262030582,
        -0.002431235632620156,
        118.32792953849406,
        0.0003713553455663881,
        62.10500925390422,
        -368.39066135682094,
        -0.0025126784870662354,
        -306.5011074516508,
        0.0031153319021852573,
        -156.84468164852024,
        -54.958677005925686,
    ];
    assert_eq!(got, want);
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
            infile: "testfiles/h2o.in",
            fort15: "testfiles/fort.15",
            fort30: "testfiles/fort.30",
            fort40: "testfiles/fort.40",
            want_e0: 20.057563725859055,
            want_fund: vec![3753.166, 3656.537, 1598.516],
            e_eps: 6e-8,
        },
        Test {
            infile: "testfiles/h2co.in",
            fort15: "testfiles/h2co.15",
            fort30: "testfiles/h2co.30",
            fort40: "testfiles/h2co.40",
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
            infile: "testfiles/c3h2.in",
            fort15: "testfiles/c3h2.15",
            fort30: "testfiles/c3h2.30",
            fort40: "testfiles/c3h2.40",
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
        let fc2 = s.rot2nd(fc2, s.axes);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, s.n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = s.make_lx(s.n3n, &sqm, &lxm);
        let (zmat, _biga, _wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
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
        let moments = s.geom.principal_moments();
        let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
        let (xcnst, e0) = xcalc(
            s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon, &fermi1, &fermi2,
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
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, _biga, _wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let f4x = load_fc4("testfiles/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
    let moments = s.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let (xcnst, e0) =
        xcalc(s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon, &[], &[]);
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
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, _biga, wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let moments = s.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let primat = s.geom.principal_moments();
    let got = alpha(s.nvib, &rotcon, &freq, &wila, &primat, &zmat, &f3qcm);
    let want = dmatrix![
       -1.1564648277177876, -0.68900933871743675, 2.5983329447479688;
    -0.09868782762986765, -0.21931495793096034, 0.16185995232026804;
    -0.14297483172390801, -0.17703802340977196, -0.14639237714841669;
                                                                         ];
    assert_abs_diff_eq!(got, want.transpose(), epsilon = 3e-6);
}

#[test]
fn test_alphaa() {
    let s = Spectro::load("testfiles/h2o.in");
    let fc2 = load_fc2("testfiles/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2, s.axes);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, s.n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let (zmat, _biga, wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", s.n3n);
    let mut f3x = s.rot3rd(f3x, s.axes);
    let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
    let f4x = load_fc4("testfiles/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    let f4qcm = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
    let moments = s.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let (xcnst, _e0) =
        xcalc(s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon, &[], &[]);
    let fund = funds(&freq, s.nvib, &xcnst);
    let Restst {
        coriolis: _,
        fermi1: _,
        fermi2: _,
        darling: _,
        i1sts,
        i1mode,
    } = s.restst(&zmat, &f3qcm, &freq);
    let got = s.alphaa(
        s.nvib, &rotcon, &freq, &wila, &zmat, &f3qcm, &fund, &i1mode, &i1sts,
    );
    let want = dmatrix![
    27.657417987118755, 14.498766626639174, 9.2673038449583238;
     26.500953159400968, 14.400078799009306, 9.1243290132344157;
     26.968408648401319, 14.279451668708212, 9.0902658215485506;
     30.255750931866725, 14.660626578959441, 9.1209114678099059;
    ];
    assert_abs_diff_eq!(got, want, epsilon = 2e-5);
}

#[test]
fn restst() {
    struct Test {
        infile: &'static str,
        fort15: &'static str,
        fort30: &'static str,
        want: Restst,
    }
    let tests = [
        Test {
            infile: "testfiles/h2o.in",
            fort15: "testfiles/fort.15",
            fort30: "testfiles/fort.30",
            want: Restst {
                coriolis: vec![],
                fermi1: vec![],
                fermi2: vec![],
                darling: vec![Darling::new(1, 0)],
                i1sts: vec![
                    vec![0, 0, 0],
                    vec![1, 0, 0],
                    vec![0, 1, 0],
                    vec![0, 0, 1],
                    vec![2, 0, 0],
                    vec![0, 2, 0],
                    vec![0, 0, 2],
                    vec![1, 1, 0],
                    vec![1, 0, 1],
                    vec![0, 1, 1],
                ],
                i1mode: vec![0, 1, 2],
            },
        },
        Test {
            infile: "testfiles/h2co.in",
            fort15: "testfiles/h2co.15",
            fort30: "testfiles/h2co.30",
            want: Restst {
                coriolis: vec![Coriolis::new(5, 4)],
                fermi1: vec![Fermi1::new(3, 1)],
                fermi2: vec![Fermi2::new(4, 2, 0)],
                darling: vec![Darling::new(1, 0), Darling::new(5, 4)],
                i1sts: vec![
                    vec![0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1],
                    vec![2, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, 0, 0, 0],
                    vec![0, 0, 2, 0, 0, 0],
                    vec![0, 0, 0, 2, 0, 0],
                    vec![0, 0, 0, 0, 2, 0],
                    vec![0, 0, 0, 0, 0, 2],
                    vec![1, 1, 0, 0, 0, 0],
                    vec![1, 0, 1, 0, 0, 0],
                    vec![1, 0, 0, 1, 0, 0],
                    vec![1, 0, 0, 0, 1, 0],
                    vec![1, 0, 0, 0, 0, 1],
                    vec![0, 1, 1, 0, 0, 0],
                    vec![0, 1, 0, 1, 0, 0],
                    vec![0, 1, 0, 0, 1, 0],
                    vec![0, 1, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 0, 0],
                    vec![0, 0, 1, 0, 1, 0],
                    vec![0, 0, 1, 0, 0, 1],
                    vec![0, 0, 0, 1, 1, 0],
                    vec![0, 0, 0, 1, 0, 1],
                    vec![0, 0, 0, 0, 1, 1],
                ],
                i1mode: vec![0, 1, 2, 3, 4, 5],
            },
        },
        Test {
            infile: "testfiles/c3h2.in",
            fort15: "testfiles/c3h2.15",
            fort30: "testfiles/c3h2.30",
            want: Restst {
                coriolis: vec![
                    Coriolis::new(5, 4),
                    Coriolis::new(6, 5),
                    Coriolis::new(7, 4),
                    Coriolis::new(7, 5),
                    Coriolis::new(7, 6),
                    Coriolis::new(8, 6),
                    Coriolis::new(8, 7),
                ],
                fermi1: vec![
                    Fermi1::new(2, 0),
                    Fermi1::new(6, 2),
                    Fermi1::new(7, 2),
                    Fermi1::new(8, 2),
                ],
                fermi2: vec![],
                darling: vec![
                    Darling::new(1, 0),
                    Darling::new(5, 4),
                    Darling::new(6, 5),
                    Darling::new(7, 5),
                    Darling::new(7, 6),
                    Darling::new(8, 6),
                    Darling::new(8, 7),
                ],
                i1sts: vec![
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![2, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 2, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 2, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 2, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 2, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 2, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 2, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 2],
                    vec![1, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 1, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 1, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 1, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 1, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 1],
                ],
                i1mode: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            },
        },
    ];
    for test in tests {
        let s = Spectro::load(test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2, s.axes);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, s.n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = s.make_lx(s.n3n, &sqm, &lxm);
        let (zmat, _biga, _wila) = s.zeta(s.natom, s.nvib, &lxm, &w);
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x, s.axes);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq, s.i3vib);
        let got = s.restst(&zmat, &f3qcm, &freq);
        assert_eq!(got.coriolis, test.want.coriolis);
        assert_eq!(got.fermi1, test.want.fermi1);
        assert_eq!(got.fermi2, test.want.fermi2);
        assert_eq!(got.darling, test.want.darling);
        assert_eq!(got.i1sts, test.want.i1sts);
        assert_eq!(got.i1mode, test.want.i1mode);
        assert_eq!(got, test.want);
    }
}
