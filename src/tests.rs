use std::str::FromStr;

use approx::assert_abs_diff_eq;
use symm::Molecule;

use crate::Curvil::*;
use crate::*;

use na::{dmatrix, dvector};
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
 C     -0.00000000        1.69593625        0.00000000
 C      1.27805254        4.09067473        0.00000000
 C     -1.27805254        4.09067473        0.00000000
 H      2.94342058        5.18191475       -0.00000000
 H     -2.94342058        5.18191475        0.00000000",
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
        is_linear: None,
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
 He      0.0000000000        0.0000000000       -3.3410028680
 H      0.0000000000        0.0000000000       -1.0030394677
 H      0.0000000000        0.0000000000        1.0030394709
 He      0.0000000000        0.0000000000        3.3410028648
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
        is_linear: None,
    };
    assert_eq!(got.curvils, want.curvils);
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
    let axes = spectro.geom.reorder();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let got = spectro.rot2nd(fc2, axes);
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
    let axes = spectro.geom.reorder();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = spectro.rot2nd(fc2, axes);
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
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();
    spectro.geom.normalize();
    let axes = spectro.geom.reorder();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let n3n = 3 * spectro.natoms();
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
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
    let spectro = Spectro::load("testfiles/h2o.in");
    spectro.run();
}

#[test]
fn test_zeta() {
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();

    spectro.geom.normalize();
    let axes = spectro.geom.reorder();

    let rotor = spectro.rotor_type();
    let natom = spectro.natoms();
    let n3n = 3 * natom;
    let nvib = n3n - 6
        + if let Rotor::Linear = rotor {
            spectro.is_linear = Some(true);
            1
        } else {
            spectro.is_linear = Some(false);
            0
        };
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (_harms, lxm) = symm_eigen_decomp(fxm);
    // TODO test the other return values
    let (_zmat, _biga, got) = spectro.zeta(natom, nvib, &lxm, &w);
    // had to swap sign of first row since my LXM has a different sign
    let want = dmatrix![
     -3.6123785074337889e-08,      -1.2692098865561809,  -6.6462962133861936e-08,   -2.9957242484449652e-09,  1.8035956714004908e-09,  -1.0258674709717752e-07 ;
        -0.9155081475600737  ,4.7344317466446739e-08  ,   -1.7484950796657917  ,-2.9480138066186021e-09  ,  2.318919812342448e-09  ,   -2.6640032272258658 ;
        -1.2781035077371032  ,2.1818979178966913e-08  ,    1.2524505892097131  ,-2.2198838796685475e-08  , 1.7461636243495606e-08  ,  -0.02565291852739008 ;
    ];
    assert_abs_diff_eq!(got, want, epsilon = 1e-6);
}

