//! calculate anharmonic constants for symmetric tops

type Tensor3 = tensor::tensor3::Tensor3<f64>;

use crate::{
    f3qcm::F3qcm,
    f4qcm::F4qcm,
    ifrm1::Ifrm1,
    ifrm2::Ifrm2,
    mode::Mode,
    resonance::{Fermi1, Fermi2},
    utils::make_e0,
    Dmat, Dvec, Spectro,
};

/// make the second component of E0 for symmetric tops
fn make_e2(
    modes: &[Mode],
    freq: &Dvec,
    f4qcm: &F4qcm,
    f3qcm: &F3qcm,
    ifrm1: &Ifrm1,
) -> f64 {
    let (n1dm, n2dm, _) = Mode::count(modes);
    let (i1mode, i2mode, _) = Mode::partition(modes);
    // e2
    // sss and ssss terms
    let mut f4s = 0.0;
    let mut f3s = 0.0;
    let mut f3kss = 0.0;
    for kk in 0..n2dm {
        let (k, _) = i2mode[kk];
        let wk = freq[k].powi(2);
        f4s += f4qcm[(k, k, k, k)] / 48.0;
        f3s += 11.0 * f3qcm[(k, k, k)].powi(2) / (freq[k] * 144.0);

        // kss and fss terms
        for ll in 0..n1dm {
            let l = i1mode[ll];
            let zval4 = f3qcm[(k, k, l)].powi(2);
            let wl = freq[l].powi(2);
            if ifrm1.check(k, l) {
                let delta4 = 2.0 * (2.0 * freq[k] + freq[l]);
                f3kss += zval4 / (16.0 * delta4);
            } else {
                let delta4 = 4.0 * wk - wl;
                f3kss += zval4 * freq[l] / (16.0 * delta4);
            }
        }
    }
    f4s + f3s + f3kss
}

/// make the third component of E0 for symmetric tops
fn make_e3(
    modes: &[Mode],
    freq: &Dvec,
    f3qcm: &F3qcm,
    ifrm1: &Ifrm1,
    ifrm2: &Ifrm2,
    ifrmchk: &tensor::Tensor3<usize>,
) -> f64 {
    let (n1dm, n2dm, _) = Mode::count(modes);
    let (i1mode, i2mode, _) = Mode::partition(modes);
    let mut f3kst = 0.0;
    let mut f3sst = 0.0;
    let mut f3stu = 0.0;
    // they check that ilin == 0 again here, but this all should only be called
    // if the molecule isn't linear
    for kk in 0..n2dm {
        let (k, _) = i2mode[kk];
        let wk = freq[k].powi(2);
        for ll in 0..n2dm {
            let (l, _) = i2mode[ll];
            if k == l {
                continue;
            }
            let wl = freq[l].powi(2);
            let zval5 = f3qcm[(k, k, l)].powi(2);
            if ifrm1.check(k, l) {
                // very encouraging comment from jan martin here: "this is
                // completely hosed!!!! dimension is cm-1 and should be cm !!!
                // perhaps he's multiplying by 2*FREQ(L) when he should divide?"
                // I'm assuming he fixed this and the code I've translated is
                // right.
                let delta5 =
                    (8.0 * wk + wl) / (2.0 * freq[k] + freq[l]) / (2.0 * wl);
                f3sst += zval5 * delta5 / 16.0;
            } else {
                let delta5 = (8.0 * wk + wl) / (freq[l] * (4.0 * wk - wl));
                f3sst += zval5 * delta5 / 16.0;
            }

            for mm in 0..n1dm {
                let m = i1mode[mm];
                if k <= l {
                    f3kst = 0.0;
                } else {
                    let zval6 = f3qcm[(k, l, m)].powi(2);
                    let xkst = freq[(k)] * freq[(l)] * freq[(m)];
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = freq[(k)] - freq[(l)] - freq[(m)];
                    if ifrm2.check((k, l), m) {
                        let delta6 = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else if ifrm2.check((l, m), k) {
                        let delta6 = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else if ifrm2.check((k, m), l) {
                        let delta6 = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else {
                        let delta6 = d1 * d2 * d3 * d4;
                        f3kst -= zval6 * xkst / (2.0 * delta6);
                    }
                }
            }

            // stu term
            for mm in 0..n2dm {
                let (m, _) = i2mode[mm];
                if k <= l || l <= m {
                    f3stu = 0.0;
                } else {
                    let zval7 = f3qcm[(k, l, m)].powi(2);
                    let xstu = freq[(k)] * freq[(l)] * freq[(m)];
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = freq[(k)] - freq[(l)] - freq[(m)];
                    // TODO these should be ifrm2 as well, I think, but we
                    // haven't had any issues here yet
                    if ifrmchk[(k, l, m)] != 0 {
                        let delta7 = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        let delta7 = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta7 = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else {
                        let delta7 = d1 * d2 * d3 * d4;
                        f3stu -= 2.0 * zval7 * xstu / (delta7);
                    }
                }
            }
        }
    }
    f3kst + f3sst + f3stu
}

impl Spectro {
    /// degenerate-degenerate interactions for anharmonic constants of
    /// symmetric tops
    fn deg_deg(
        &self,
        i2mode: &Vec<(usize, usize)>,
        f4qcm: &F4qcm,
        freq: &Dvec,
        i1mode: &Vec<usize>,
        f3qcm: &F3qcm,
        ifrm1: &Ifrm1,
        xcnst: &mut Dmat,
        n2dm: usize,
        ifrm2: &Ifrm2,
        ia: usize,
        zmat: &tensor::Tensor3<f64>,
        ib: usize,
        ixyz: usize,
    ) {
        deg_deg1(i2mode, f4qcm, freq, i1mode, f3qcm, ifrm1, xcnst);

        for kk in 1..n2dm {
            let k = i2mode[kk].0;
            // might be -1
            for ll in 0..kk {
                let (l, l2) = i2mode[ll];
                let val1 = (f4qcm[(k, k, l, l)] + f4qcm[(k, k, l2, l2)]) / 8.0;

                let val2: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        -(f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4.0 * freq[m]))
                    })
                    .sum();

                let valu: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        if ifrm2.check((k, l), m) {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else if ifrm2.check((l, m), k) {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else if ifrm2.check((k, m), l) {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)].powi(2)
                                - freq[(k)].powi(2)
                                - freq[(l)].powi(2);
                            -0.25
                                * (f3qcm[(k, l, m)].powi(2))
                                * freq[(m)]
                                * val3
                                / delta
                        }
                    })
                    .sum();

                let valus: f64 = i2mode
                    .iter()
                    .map(|&(m, _)| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrm2.check((l, m), k) {
                            let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrm2.check((k, m), l) {
                            let delta = 8.0 * (2.0 * freq[(k)] + freq[(l)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrm2.check((k, l), m) {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrm2.check((l, m), k) {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrm2.check((k, m), l) {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)].powi(2)
                                - freq[(k)].powi(2)
                                - freq[(l)].powi(2);
                            -0.5 * (f3qcm[(klm)].powi(2)) * freq[(m)] * val3
                                / delta
                        }
                    })
                    .sum();

                let val5 = freq[(k)] / freq[(l)];
                let val6 = freq[(l)] / freq[(k)];
                let val7 = 0.5 * self.rotcon[(ia)] * (zmat[(k, l2, 2)].powi(2))
                    + self.rotcon[(ib)] * (zmat[(k, l, ixyz)].powi(2));
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + valus + val8;
                let k2 = i2mode[kk].1;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
                xcnst[(k2, l2)] = value;
                xcnst[(l2, k2)] = value;
            }
        }
    }

    /// g constants for degenerate modes
    fn make_gcnst(
        &self,
        n2dm: usize,
        i2mode: Vec<(usize, usize)>,
        f4qcm: &F4qcm,
        freq: &Dvec,
        i1mode: Vec<usize>,
        f3qcm: &F3qcm,
        ifrm1: Ifrm1,
        ia: usize,
        zmat: &tensor::Tensor3<f64>,
        ifrmchk: tensor::Tensor3<usize>,
        ib: usize,
        ixyz: usize,
    ) -> Dmat {
        let mut gcnst = Dmat::zeros(self.nvib, self.nvib);
        for kk in 0..n2dm {
            let (k, k2) = i2mode[kk];
            let val1 = -f4qcm[(k, k, k, k)] / 48.0;
            let wk = freq[k].powi(2);

            let valu: f64 = i1mode
                .iter()
                .map(|&l| {
                    let val2 = f3qcm[(k, k, l)].powi(2);
                    if ifrm1.check(k, l) {
                        let val3 = 32.0 * (2.0 * freq[(k)] + freq[(l)]);
                        val2 / val3
                    } else {
                        let wl = freq[(l)] * freq[(l)];
                        let val3 = val2 * freq[(l)];
                        let val4 = 16.0 * (4.0 * wk - wl);
                        -val3 / val4
                    }
                })
                .sum();

            let valus: f64 = i2mode
                .iter()
                .map(|&(l, _)| {
                    let val2 = f3qcm[(k, k, l)].powi(2);
                    if ifrm1.check(k, l) {
                        let val3 = 1.0 / (8.0 * freq[(l)]);
                        let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                        -val2 * (val3 + val4)
                    } else {
                        let wl = freq[(l)] * freq[(l)];
                        let val3 = 8.0 * wk - wl;
                        let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                        val2 * val3 / val4
                    }
                })
                .sum();

            let val7 = self.rotcon[(ia)] * (zmat[(k, k2, 2)].powi(2));
            let value = val1 + valu + valus + val7;
            gcnst[(k, k)] = value;
            gcnst[(k2, k2)] = value;
        }
        for kk in 1..n2dm {
            let (k, k2) = i2mode[kk];
            for ll in 0..kk {
                let (l, l222) = i2mode[ll];

                let valu: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)] * freq[(k)] * freq[(l)];
                            0.5 * (f3qcm[(klm)].powi(2)) * val3 / delta
                        }
                    })
                    .sum();

                let valus: f64 = i2mode
                    .iter()
                    .map(|&(m, _)| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrmchk[(l, m, k)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(k)] + freq[(l)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)] * freq[(k)] * freq[(l)];
                            -(f3qcm[(klm)].powi(2)) * val3 / delta
                        }
                    })
                    .sum();

                let val7 =
                    -2.0 * self.rotcon[(ib)] * (zmat[(k, l, ixyz)].powi(2))
                        + self.rotcon[(ia)] * (zmat[(k, l222, 2)].powi(2))
                        + 2.0
                            * self.rotcon[(ia)]
                            * zmat[(k, k2, 2)]
                            * zmat[(l, l222, 2)];
                // NOTE last two zmat values are the same sign instead of
                // opposite for nh3, causes an issue in gcnst. hopefully it will
                // cause small issues later on.. geometry looks okay, just
                // pointing the opposite direction on the z axis from the
                // fortran version. an element with the other sign exists in my
                // zmat, but I don't really want to start randomly changing
                // stuff without more tests. If I see multiple cases like this,
                // I will feel more comfortable changing the index in the code
                let value = valu + valus + val7;
                gcnst[(k, l)] = value;
                gcnst[(l, k)] = value;
                gcnst[(k2, l222)] = value;
                gcnst[(l222, k2)] = value;
            }
        }
        gcnst
    }

    /// nondeg-deg interactions for anharmonic constants of a symmetric top
    pub(crate) fn nondeg_deg(
        &self,
        i1mode: &Vec<usize>,
        i2mode: &Vec<(usize, usize)>,
        f4qcm: &F4qcm,
        f3qcm: &F3qcm,
        freq: &Dvec,
        ifrm2: &Ifrm2,
        ib: usize,
        zmat: &tensor::Tensor3<f64>,
        xcnst: &mut Dmat,
    ) {
        for &k in i1mode {
            for &(l, l2) in i2mode {
                let val1 = f4qcm[(k, k, l, l)] / 4.0;

                let mut val2 = 0.0;
                for &m in i1mode {
                    val2 -=
                        f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4. * freq[m]);
                }

                let mut valu = 0.0;
                for &(m, _) in i2mode {
                    let klm = (k, l, m);
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                    if ifrm2.check((l, m), k) {
                        let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                        valu -= (f3qcm[(klm)].powi(2)) / delta;
                    } else if ifrm2.check((k, l), m) {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else if ifrm2.check((l, m), k) {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else if ifrm2.check((k, m), l) {
                        let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else {
                        let delta = -d1 * d2 * d3 * d4;
                        let val3 = freq[(m)].powi(2)
                            - freq[(k)].powi(2)
                            - freq[(l)].powi(2);
                        valu -= 0.5 * (f3qcm[(klm)].powi(2)) * freq[(m)] * val3
                            / delta;
                    }
                }
                let val5 = freq[(k)] / freq[(l)];
                let val6 = freq[(l)] / freq[(k)];
                let val7 = self.rotcon[(ib)]
                    * (zmat[(k, l, 0)].powi(2) + zmat[(k, l, 1)].powi(2));
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + val8;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
                xcnst[(k, l2)] = value;
                xcnst[(l2, k)] = value;
            }
        }
    }

    /// calculate the nondeg-nondeg anharmonic constants
    pub(crate) fn nondeg_nondeg(
        &self,
        i1mode: &Vec<usize>,
        f4qcm: &F4qcm,
        f3qcm: &F3qcm,
        freq: &Dvec,
        ifrm2: &Ifrm2,
        ia: usize,
        zmat: &tensor::Tensor3<f64>,
        ifrm1: &Ifrm1,
        xcnst: &mut Dmat,
    ) {
        for &k in i1mode {
            let kkkk = (k, k, k, k);
            let val1 = f4qcm[kkkk] / 16.0;
            let wk = freq[k].powi(2);

            let mut valu = 0.0;
            for &l in i1mode {
                let val2 = f3qcm[(k, k, l)].powi(2);
                if ifrm1.check(k, l) {
                    let val3 = 1.0 / (8.0 * freq[l]);
                    let val4 = 1.0 / (32.0 * (2.0 * freq[k] + freq[l]));
                    valu -= val2 * (val3 + val4);
                } else {
                    let wl = freq[l].powi(2);
                    let val3 = 8.0 * wk - 3.0 * wl;
                    let val4 = 16.0 * freq[l] * (4.0 * wk - wl);
                    valu -= val2 * val3 / val4;
                }
            }
            xcnst[(k, k)] = val1 + valu;
        }

        // pretty sure it's safe to ignore this because the next loop should
        // never be entered for a diatomic since there should only be one mode
        if self.rotor.is_diatomic() {
            todo!("goto funds section. return?")
        }

        let n1dm = i1mode.len();
        for kk in 1..n1dm {
            let k = i1mode[kk];
            // might be kk-1 not sure
            for ll in 0..kk {
                let l = i1mode[ll];
                let kkll = (k, k, l, l);
                let val1 = f4qcm[kkll] / 4.0;

                let mut val2 = 0.0;
                for &m in i1mode {
                    val2 -=
                        f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4.0 * freq[m]);
                }

                let mut valu = 0.0;
                for &m in i1mode {
                    let klm = (k, m, l);
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                    if ifrm2.check((l, m), k) {
                        let delta = 8.0 * (2.0 * freq[l] + freq[k]);
                        valu -= (f3qcm[klm].powi(2)) / delta;
                    } else if ifrm2.check((k, m), l) {
                        let delta = 8.0 * (2.0 * freq[k] + freq[l]);
                        valu -= f3qcm[klm].powi(2) / delta;
                    } else if ifrm2.check((k, l), m) {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else if ifrm2.check((l, m), k) {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else if ifrm2.check((k, m), l) {
                        let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else {
                        let delta = -d1 * d2 * d3 * d4;
                        let val3 =
                            freq[m].powi(2) - freq[k].powi(2) - freq[l].powi(2);
                        valu -=
                            0.5 * (f3qcm[klm].powi(2)) * freq[m] * val3 / delta;
                    }
                }

                let val5 = freq[k] / freq[l];
                let val6 = freq[l] / freq[k];
                let val7 = self.rotcon[ia] * zmat[(k, l, 2)].powi(2);
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + val8;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
            }
        }
    }

    /// compute `ia`, `ib`, `n2dm`, `i1mode`, `i2mode`, and `ixyz` for [xcals]
    pub(crate) fn setup_xcals(
        &self,
        modes: &[Mode],
        wila: &Dmat,
    ) -> (usize, usize, usize, Vec<usize>, Vec<(usize, usize)>, usize) {
        let (ia, ib) = (2, 1);
        let (_, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        // find out which of a(xz)tb or a(yz)tb are zero
        let (ixyz, _ia1, _ia2, _ix, _iy) = if !self.rotor.is_linear() {
            const TOL: f64 = 0.000001;
            let mut ixz = 0;
            let mut iyz = 0;
            for &(_, i2) in &i2mode {
                if wila[(i2, 3)].abs() <= TOL {
                    ixz += 1;
                }
                if wila[(i2, 4)].abs() <= TOL {
                    iyz += 1;
                }
            }
            if ixz > 0 && iyz > 0 {
                (1, 2, 3, 0, 1)
            } else if ixz > 0 {
                (1, 2, 3, 0, 1)
            } else if iyz > 0 {
                (0, 0, 4, 1, 0)
            } else {
                panic!("big problem in xcals");
            }
        } else {
            // actually everything I have here pretty much assumes the molecule
            // is not linear, so good to have this todo here
            todo!()
        };
        (ia, ib, n2dm, i1mode, i2mode, ixyz)
    }

    /// calculate the anharmonic constants and E_0 for a symmetric top
    pub fn xcals(
        &self,
        f4qcm: &F4qcm,
        freq: &Dvec,
        f3qcm: &F3qcm,
        zmat: &Tensor3,
        fermi1: &[Fermi1],
        fermi2: &[Fermi2],
        modes: &[Mode],
        wila: &Dmat,
    ) -> (Dmat, Dmat, f64) {
        let (ia, ib, n2dm, i1mode, i2mode, ixyz) =
            self.setup_xcals(modes, wila);
        // NOTE skipping zeta checks, but they only print stuff

        let (ifrmchk, ifrm1, ifrm2) = self.make_fermi_checks(fermi1, fermi2);

        let e1 = make_e0(modes, f4qcm, f3qcm, freq, &ifrm1, &ifrmchk);
        let e2 = make_e2(modes, freq, f4qcm, f3qcm, &ifrm1);
        let e3 = make_e3(modes, freq, f3qcm, &ifrm1, &ifrm2, &ifrmchk);
        let e0 = e1 + e2 + e3;

        // start calculating anharmonic constants
        let mut xcnst = Dmat::zeros(self.nvib, self.nvib);

        self.nondeg_nondeg(
            &i1mode, f4qcm, f3qcm, freq, &ifrm2, ia, zmat, &ifrm1, &mut xcnst,
        );

        self.nondeg_deg(
            &i1mode, &i2mode, f4qcm, f3qcm, freq, &ifrm2, ib, zmat, &mut xcnst,
        );

        self.deg_deg(
            &i2mode, f4qcm, freq, &i1mode, f3qcm, &ifrm1, &mut xcnst, n2dm,
            &ifrm2, ia, zmat, ib, ixyz,
        );

        let gcnst = self.make_gcnst(
            n2dm, i2mode, f4qcm, freq, i1mode, f3qcm, ifrm1, ia, zmat, ifrmchk,
            ib, ixyz,
        );

        (xcnst, gcnst, e0)
    }
}

pub(crate) fn deg_deg1(
    i2mode: &Vec<(usize, usize)>,
    f4qcm: &F4qcm,
    freq: &Dvec,
    i1mode: &Vec<usize>,
    f3qcm: &F3qcm,
    ifrm1: &Ifrm1,
    xcnst: &mut Dmat,
) {
    for (kk, &(k, _)) in i2mode.iter().enumerate() {
        let val1 = f4qcm[(k, k, k, k)] / 16.0;
        let wk = freq[k].powi(2);

        let mut valu = 0.0;
        for &l in i1mode {
            let val2 = f3qcm[(k, k, l)].powi(2);
            if ifrm1.check(k, l) {
                let val3 = 1.0 / (8.0 * freq[(l)]);
                let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                valu -= val2 * (val3 + val4);
            } else {
                let wl = freq[(l)] * freq[(l)];
                let val3 = 8.0 * wk - 3.0 * wl;
                let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                valu -= val2 * val3 / val4;
            }
        }

        let mut valus = 0.0;
        for &(l, _) in i2mode {
            let val2 = f3qcm[(k, k, l)].powi(2);
            if ifrm1.check(k, l) {
                let val3 = 1.0 / (8.0 * freq[(l)]);
                let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                valus -= val2 * (val3 + val4);
            } else {
                let wl = freq[(l)] * freq[(l)];
                let val3 = 8.0 * wk - 3.0 * wl;
                let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                valus -= val2 * val3 / val4;
            }
        }

        let value = val1 + valu + valus;
        let k2 = i2mode[kk].1;
        xcnst[(k, k)] = value;
        xcnst[(k2, k2)] = value;
    }
}
