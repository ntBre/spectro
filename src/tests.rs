use std::str::FromStr;

use symm::Molecule;

use crate::{Dummy, DummyVal, Spectro};

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
            vec![2, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 4],
            vec![3, 5],
            vec![2, 4, 1],
            vec![3, 5, 1],
            vec![4, 2, 1, 3],
            vec![5, 3, 1, 2],
        ],
        degmodes: vec![],
        dummies: vec![],
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
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![2, 1, 3],
            vec![3, 2, 4],
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
    };
    assert_eq!(got.curvils, want.curvils);
    assert_eq!(got, want);
}
