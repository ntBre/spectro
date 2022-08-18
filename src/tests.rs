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

#[test]
fn test_force3() {
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
    let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let f3x = load_fc3("testfiles/fort.30", n3n);
    let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
    let got = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
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
    let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let f4x = load_fc4("testfiles/fort.40", n3n);
    let mut f4x = spectro.rot4th(n3n, natom, f4x, axes);
    let got = force4(n3n, &mut f4x, &lx, nvib, &freq, i4vib);
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
    let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let (zmat, _biga, _wila) = spectro.zeta(natom, nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", n3n);
    let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
    let f3qcm = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
    let f4x = load_fc4("testfiles/fort.40", n3n);
    let mut f4x = spectro.rot4th(n3n, natom, f4x, axes);
    let f4qcm = force4(n3n, &mut f4x, &lx, nvib, &freq, i4vib);
    let moments = spectro.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let (xcnst, e0) = xcalc(nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);
    let wante0 = 20.057563725859055;
    assert_abs_diff_eq!(e0, wante0, epsilon = 6e-8);
    let got = funds(&freq, nvib, &xcnst);
    let want = vec![3753.166, 3656.537, 1598.516];
    assert_abs_diff_eq!(Dvec::from(got), Dvec::from(want), epsilon = 1e-3);
}

#[test]
fn test_enrgy() {
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
    let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let (zmat, _biga, _wila) = spectro.zeta(natom, nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", n3n);
    let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
    let f3qcm = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
    let f4x = load_fc4("testfiles/fort.40", n3n);
    let mut f4x = spectro.rot4th(n3n, natom, f4x, axes);
    let f4qcm = force4(n3n, &mut f4x, &lx, nvib, &freq, i4vib);
    let moments = spectro.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let (xcnst, e0) = xcalc(nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);
    let wante0 = 20.057563725859055;
    assert_abs_diff_eq!(e0, wante0, epsilon = 6e-8);
    let fund = funds(&freq, nvib, &xcnst);
    let i1sts = vec![
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
    ];
    let got = enrgy(&fund, &freq, &xcnst, e0, &i1sts, &[0, 1, 2]);
    // my numbers after comparing visually to fortran
    let want = vec![
        (4656.438188555293, vec![0, 0, 0]),
        (6254.953686350812, vec![0, 0, 1]),
        (7818.958032793332, vec![0, 0, 2]),
        (8312.975664427879, vec![0, 1, 0]),
        (8409.60462543482, vec![1, 0, 0]),
        (9895.66935813587, vec![0, 1, 1]),
        (9988.129295095654, vec![1, 0, 1]),
        (11883.428277109273, vec![0, 2, 0]),
        (11899.804775421886, vec![1, 1, 0]),
        (12065.411263606182, vec![2, 0, 0]),
    ];
    assert_eq!(got, want);
}

#[test]
fn test_alpha() {
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
    let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let (zmat, _biga, wila) = spectro.zeta(natom, nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", n3n);
    let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
    let f3qcm = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
    let moments = spectro.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let primat = spectro.geom.principal_moments();
    let got = alpha(nvib, &rotcon, &freq, &wila, &primat, &zmat, &f3qcm);
    let want = dmatrix![
       -1.1564648277177876, -0.68900933871743675, 2.5983329447479688;
    -0.09868782762986765, -0.21931495793096034, 0.16185995232026804;
    -0.14297483172390801, -0.17703802340977196, -0.14639237714841669;
                                                                         ];
    assert_abs_diff_eq!(got, want.transpose(), epsilon = 3e-6);
}

#[test]
fn test_alphaa() {
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
    let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
    let fc2 = load_fc2("testfiles/fort.15", n3n);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let w = spectro.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = spectro.form_sec(fc2, n3n, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = spectro.make_lx(n3n, &sqm, &lxm);
    let (zmat, _biga, wila) = spectro.zeta(natom, nvib, &lxm, &w);
    let f3x = load_fc3("testfiles/fort.30", n3n);
    let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
    let f3qcm = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
    let f4x = load_fc4("testfiles/fort.40", n3n);
    let mut f4x = spectro.rot4th(n3n, natom, f4x, axes);
    let f4qcm = force4(n3n, &mut f4x, &lx, nvib, &freq, i4vib);
    let moments = spectro.geom.principal_moments();
    let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
    let (xcnst, _e0) = xcalc(nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);
    let fund = funds(&freq, nvib, &xcnst);
    let i1sts = vec![
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
    ];
    let got = spectro.alphaa(
        nvib,
        &rotcon,
        &freq,
        &wila,
        &zmat,
        &f3qcm,
        &fund,
        &[0, 1, 2],
        &i1sts,
    );
    let want = dmatrix![
    27.657417987118755, 14.498766626639174, 9.2673038449583238;
     26.500953159400968, 14.400078799009306, 9.1243290132344157;
     26.968408648401319, 14.279451668708212, 9.0902658215485506;
     30.255750931866725, 14.660626578959441, 9.1209114678099059;
    ];
    assert_abs_diff_eq!(got, want, epsilon = 2e-5);
}
