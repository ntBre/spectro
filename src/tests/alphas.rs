use std::collections::HashMap;

use nalgebra::dvector;

use crate::{f3qcm::F3qcm, utils::load_vec, Spectro, Tensor3};

use super::{check_mat, load_dmat};

#[test]
fn make_alpha() {
    let s = Spectro {
        rotcon: vec![
            4.5034193376213194,
            4.5034192814538905,
            3.9601029298042021,
        ],
        primat: vec![
            3.7432957395278987,
            3.7432957862149348,
            4.2568667326682146,
        ],
        nvib: 6,
        ..Spectro::default()
    };
    let n1dm = 2;
    let n2dm = 2;
    let i1mode = vec![2, 5];
    let i2mode = vec![(0, 1), (3, 4)];
    let ia = 2;
    let ib = 1;
    let ic = 0;
    let iaia = 5;
    let iaib = 4;
    let ibib = 2;
    let ibic = 1;
    let freq = dvector![
        2437.0024382429601,
        2437.0020300169222,
        2428.1546959531433,
        1145.7614713948005,
        1145.7567008646774,
        1015.4828482214278,
        0.046516934814435321,
        0.025278773879589642,
        0.015292949295790762,
        0.0077266799336841666,
        0.011089653435072953,
        0.014380890514910207
    ];
    let zmat = Tensor3::load("testfiles/ph3/zmat");
    let wila = load_dmat("testfiles/ph3/wila", 6, 6);
    let f3qcm = F3qcm::new(load_vec("testfiles/ph3/f3qcm"));
    let icorol = HashMap::from([((3, 5), 1), ((5, 3), 1)]);
    let got = s.make_alpha(
        n1dm, &i1mode, ia, &freq, &wila, iaia, icorol, &zmat, &f3qcm, n2dm,
        &i2mode, iaib, ib, ibib, ibic, ic,
    );
    let want = load_dmat("testfiles/ph3/alpha", 6, 3);

    check_mat(&got, &want, 1e-10, "ph3");
}
