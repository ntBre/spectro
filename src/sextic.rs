use std::cmp::{max, min};

use tensor::{Tensor3, Tensor4};

use crate::{
    utils::{find3r, ioff, make_tau},
    Dmat, Dvec, Spectro, Vec3, SQLAM,
};

/// struct holding the sextic distortion constants
pub(crate) struct Sextic {}

#[allow(unused)]
impl Sextic {
    pub(crate) fn new(
        spectro: &Spectro,
        wila: &Dmat,
        zmat: &Tensor3,
        freq: &Dvec,
        f3qcm: &[f64],
        rotcon: &[f64],
    ) -> Self {
        let nvib = spectro.nvib;
        // convert to Hz from cm⁻¹
        const CONST2: f64 = 2.99792458e10;
        let maxcor = if spectro.is_linear() { 2 } else { 3 };
        let primat = spectro.geom.principal_moments();
        let c = c_mat(maxcor, nvib, freq, &primat, wila);
        let t = t_mat(maxcor, nvib, freq, &c);
        let cc = cc_tensor(nvib, maxcor, freq, &c, zmat, rotcon);
        let tau = make_tau(maxcor, nvib, freq, &primat, wila);
        let scc = scc(maxcor, tau, rotcon, nvib, freq, cc, f3qcm, &c, spectro);
        Sextic {}
    }
}

fn scc(
    maxcor: usize,
    tau: Tensor4,
    rotcon: &[f64],
    nvib: usize,
    freq: &Dvec,
    cc: Tensor4,
    f3qcm: &[f64],
    c: &Dmat,
    spectro: &Spectro,
) -> Tensor3 {
    // some kind of tolerance for messing with certain values
    const TOL: f64 = 1e-4;
    let mut scc = Tensor3::zeros(3, 3, 3);
    for ixyz in 1..=maxcor {
        let iixyz = ioff(ixyz + 1);

        let mut val1 = 0.0;
        for jxyz in 1..=maxcor {
            val1 += tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)].powi(2)
                / rotcon[jxyz - 1];
        }
        val1 = 3.0 * val1 / 16.0;

        let mut val2 = 0.0;
        for i in 0..nvib {
            val2 += freq[i] * cc[(i, ixyz - 1, ixyz - 1, ixyz - 1)].powi(2);
        }
        val2 *= 2.0;

        let mut val3 = 0.0;
        for i in 0..nvib {
            for j in 0..nvib {
                for k in 0..nvib {
                    let ijk = find3r(i, j, k);
                    val3 += f3qcm[ijk]
                        * c[(i, iixyz - 1)]
                        * c[(j, iixyz - 1)]
                        * c[(k, iixyz - 1)];
                }
            }
        }
        val3 /= 6.0;

        if spectro.is_linear() {
            todo!() // goto 510
        };

        let mut val4 = 0.0;
        for jxyz in 1..=3 {
            if ixyz != jxyz {
                let div = rotcon[ixyz - 1] - rotcon[jxyz - 1];
                if div.abs() > TOL {
                    val4 += tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)]
                        .powi(2)
                        / div;
                }
            }
        }
        val4 *= 0.25;

        let value = val1 - val2 + val3 + val4;
        scc[(ixyz - 1, ixyz - 1, ixyz - 1)] = value;
    }

    if spectro.is_linear() {
        todo!() // goto 900, return?
    }

    for ixyz in 1..=3 {
        let iixyz = ioff(ixyz + 1);
        for jxyz in 1..=3 {
            if ixyz != jxyz {
                let jjxyz = ioff(jxyz + 1);
                let ijxyz = ioff(max(ixyz, jxyz)) + min(ixyz, jxyz);

                let mut vala = 0.0;
                for kxyz in 1..=3 {
                    let val1 = tau[(kxyz - 1, jxyz - 1, ixyz - 1, ixyz - 1)]
                        + 2.0 * tau[(kxyz - 1, ixyz - 1, jxyz - 1, ixyz - 1)];
                    let val1 = val1 * val1;
                    let val2 = 2.0
                        * tau[(kxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)]
                        * (tau[(kxyz - 1, ixyz - 1, jxyz - 1, jxyz - 1)]
                            + 2.0
                                * tau
                                    [(kxyz - 1, jxyz - 1, jxyz - 1, ixyz - 1)]);
                    vala += (val1 + val2) / rotcon[kxyz - 1];
                    // TODO subtract one from xyz indices
                }

                vala *= 3.0 / 32.0;

                let mut valb = 0.0;
                for i in 0..nvib {
                    let val1 = cc[(i, jxyz - 1, ixyz - 1, ixyz - 1)].powi(2)
                        + 2.0
                            * cc[(i, ixyz - 1, ixyz - 1, ixyz - 1)]
                            * cc[(i, jxyz - 1, jxyz - 1, ixyz - 1)];
                    valb += freq[i] * val1;
                }

                let mut valc = 0.0;
                for i in 0..nvib {
                    for j in 0..nvib {
                        let val1 = c[(j, iixyz - 1)] * c[(i, jjxyz - 1)]
                            + 4.0 * c[(j, ijxyz - 1)] * c[(i, ijxyz - 1)];
                        for k in 0..nvib {
                            let ijk = find3r(i, j, k);
                            valc += f3qcm[ijk] * c[(k, iixyz - 1)] * val1;
                        }
                    }
                }
                valc /= 4.0;

                let div = 8.0 * (rotcon[ixyz - 1] - rotcon[jxyz - 1]);
                let mut vald = 0.0;
                if div.abs() > TOL {
                    let val1 = 4.0
                        * tau[(jxyz - 1, jxyz - 1, jxyz - 1, ixyz - 1)]
                        - 3.0 * tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)];
                    vald += val1
                        * tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)]
                        / div;
                }

                let mut vale = 0.0;
                let mut valf = 0.0;
                for kxyz in 1..=3 {
                    if kxyz != jxyz {
                        if kxyz != ixyz {
                            let div = 4.0
                                * (rotcon[ixyz - 1] - rotcon[kxyz - 1]).powi(2);
                            if div > TOL {
                                let val1 = (rotcon[(ixyz - 1)]
                                    - rotcon[(kxyz - 1)])
                                    * (tau[(
                                        kxyz - 1,
                                        ixyz - 1,
                                        jxyz - 1,
                                        jxyz - 1,
                                    )] + 2.0e0
                                        * tau[(
                                            kxyz - 1,
                                            jxyz - 1,
                                            jxyz - 1,
                                            ixyz - 1,
                                        )]);
                                let val2 = (rotcon[(ixyz - 1)]
                                    - rotcon[(jxyz - 1)])
                                    * (tau[(
                                        kxyz - 1,
                                        kxyz - 1,
                                        kxyz - 1,
                                        ixyz - 1,
                                    )] - tau[(
                                        kxyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                    )]);
                                vale = vale
                                    + tau[(
                                        kxyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                    )] * (val1 + val2)
                                        / div;
                            }

                            let div = 8.0
                                * (rotcon[jxyz - 1] - rotcon[kxyz - 1]).powi(2);
                            if div > TOL {
                                let val1 = (rotcon[(jxyz - 1)]
                                    - rotcon[(kxyz - 1)])
                                    * (tau[(
                                        kxyz - 1,
                                        jxyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                    )] + 2.0
                                        * tau[(
                                            kxyz - 1,
                                            ixyz - 1,
                                            jxyz - 1,
                                            ixyz - 1,
                                        )]);
                                let val2 = 2.0
                                    * (rotcon[(ixyz - 1)] - rotcon[(jxyz - 1)])
                                    * (tau[(
                                        kxyz - 1,
                                        jxyz - 1,
                                        jxyz - 1,
                                        jxyz - 1,
                                    )] - tau[(
                                        kxyz - 1,
                                        kxyz - 1,
                                        kxyz - 1,
                                        jxyz - 1,
                                    )]);
                                let val3 = tau
                                    [(kxyz - 1, jxyz - 1, ixyz - 1, ixyz - 1)]
                                    + 2.0
                                        * tau[(
                                            kxyz - 1,
                                            ixyz - 1,
                                            jxyz - 1,
                                            ixyz - 1,
                                        )];
                                valf = valf + val3 * (val1 + val2) / div;
                            }
                        }
                    }
                }

                let value = vala - valb + valc + vald + vale + valf;
                scc[(jxyz - 1, ixyz - 1, ixyz - 1)] = value;
                scc[(ixyz - 1, jxyz - 1, ixyz - 1)] = value;
                scc[(ixyz - 1, ixyz - 1, jxyz - 1)] = value;
            }
        }
    } // end loop at line 397

    let mut vala = 0.0;
    for ixyz in 1..=3 {
        let val1 = tau[(ixyz - 1, 0, 1, 2)]
            + tau[(ixyz - 1, 1, 2, 0)]
            + tau[(ixyz - 1, 2, 0, 1)];
        let val1 = 2.0 * val1 * val1;
        let val2 = (tau[(ixyz - 1, 0, 1, 1)] + 2.0 * tau[(ixyz - 1, 1, 0, 1)])
            * (tau[(ixyz - 1, 0, 2, 2)] + 2.0 * tau[(ixyz - 1, 2, 0, 2)]);
        let val3 = (tau[(ixyz - 1, 1, 2, 2)] + 2.0 * tau[(ixyz - 1, 2, 1, 2)])
            * (tau[(ixyz - 1, 1, 0, 0)] + 2.0 * tau[(ixyz - 1, 0, 1, 0)]);
        let val4 = (tau[(ixyz - 1, 2, 0, 0)] + 2.0 * tau[(ixyz - 1, 0, 2, 0)])
            * (tau[(ixyz - 1, 2, 1, 1)] + 2.0 * tau[(ixyz - 1, 1, 2, 1)]);
        vala = vala + (val1 + val2 + val3 + val4) / rotcon[(ixyz - 1)];
    }
    vala = 3.0 * vala / 16.0;
    let mut valb = 0.0;
    for i in 0..nvib {
        let val1 = 2.0 * cc[(i, 2, 1, 0)].powi(2)
            + cc[(i, 1, 1, 0)] * cc[(i, 2, 2, 0)]
            + cc[(i, 2, 2, 1)] * cc[(i, 1, 0, 0)]
            + cc[(i, 2, 0, 0)] * cc[(i, 2, 1, 1)];
        valb = valb + freq[(i)] * val1;
    }
    valb = 2.0 * valb;
    let mut valc = 0.0;
    for i in 0..nvib {
        for j in 0..nvib {
            for k in 0..nvib {
                let ijk = find3r(i, j, k);
                let val1 = c[(i, 0)] * c[(j, 2)] * c[(k, 5)]
                    + 2.0 * c[(i, 0)] * c[(j, 4)] * c[(k, 4)]
                    + 2.0 * c[(i, 2)] * c[(j, 3)] * c[(k, 3)]
                    + 2.0 * c[(i, 5)] * c[(j, 1)] * c[(k, 1)]
                    + 8.0 * c[(i, 4)] * c[(j, 3)] * c[(k, 1)];
                valc = valc + f3qcm[(ijk)] * val1;
            }
        }
    }
    valc = valc * 0.5e0;
    let mut vald = 0.0;
    let val1 = 3.0 * (tau[(2, 1, 1, 1)] - tau[(2, 2, 2, 1)]);
    let val2 = (rotcon[(2)] - rotcon[(0)]) * tau[(2, 1, 1, 1)]
        + (rotcon[(0)] - rotcon[(1)]) * tau[(2, 2, 2, 1)];
    let div = 4.0 * (rotcon[(1)] - rotcon[(2)]).powi(2);
    if div > TOL {
        vald = val1 * val2 / div;
    }

    let mut vale = 0.0;
    let val1 = 3.0 * (tau[(2, 2, 2, 0)] - tau[(2, 0, 0, 0)]);
    let val2 = (rotcon[(0)] - rotcon[(1)]) * tau[(2, 2, 2, 0)]
        + (rotcon[(1)] - rotcon[(2)]) * tau[(2, 0, 0, 0)];
    let div = 4.0 * (rotcon[(2)] - rotcon[(0)]).powi(2);
    if div > TOL {
        vale = val1 * val2 / div;
    }
    let mut valf = 0.0;
    let val1 = 3.0 * (tau[(1, 0, 0, 0)] - tau[(1, 1, 1, 0)]);
    let val2 = (rotcon[(1)] - rotcon[(2)]) * tau[(1, 0, 0, 0)]
        + (rotcon[(2)] - rotcon[(0)]) * tau[(1, 1, 1, 0)];
    let div = 4.0 * (rotcon[(0)] - rotcon[(1)]).powi(2);
    if div > TOL {
        valf = val1 * val2 / div;
    }
    let value = vala - valb + valc + vald + vale + valf;
    scc[(0, 1, 2)] = value;
    scc[(0, 2, 1)] = value;
    scc[(1, 0, 2)] = value;
    scc[(1, 2, 0)] = value;
    scc[(2, 0, 1)] = value;
    scc[(2, 1, 0)] = value;

    scc
}

fn cc_tensor(
    nvib: usize,
    maxcor: usize,
    freq: &Dvec,
    c: &Dmat,
    zmat: &Tensor3,
    rotcon: &[f64],
) -> Tensor4 {
    let mut cc = Tensor4::zeros(nvib, 3, 3, 3);
    for i in 0..nvib {
        for ixyz in 1..=maxcor {
            let iixyz = ioff(ixyz + 1);

            let mut val = 0.0;
            for j in 0..nvib {
                let val1 = 1.0 / freq[j].powi(2) + 2.0 / freq[i].powi(2);
                let val2 = c[(j, iixyz - 1)]
                    * zmat[(j, i, ixyz - 1)]
                    * rotcon[ixyz - 1]
                    * freq[j].powf(1.5);
                val += val2 * val1 / freq[i].sqrt();
            }
            cc[(i, ixyz - 1, ixyz - 1, ixyz - 1)] = val;

            for jxyz in 1..=maxcor {
                if ixyz != jxyz {
                    let ijxyz = ioff(max(ixyz, jxyz)) + min(ixyz, jxyz);

                    let mut val = 0.0;
                    for j in 0..nvib {
                        let val1 =
                            1.0 / freq[j].powi(2) + 2.0 / freq[i].powi(2);
                        let val2 = freq[j].powf(1.5)
                            * (c[(j, iixyz - 1)]
                                * zmat[(j, i, jxyz - 1)]
                                * rotcon[jxyz - 1]
                                + 2.0
                                    * c[(j, ijxyz - 1)]
                                    * zmat[(j, i, ixyz - 1)]);
                        val += val2 * val1 / freq[i].sqrt();
                    }
                    cc[(i, jxyz - 1, ixyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, jxyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, ixyz - 1, jxyz - 1)] = val;

                    for kxyz in 1..=maxcor {
                        if kxyz != jxyz {
                            if kxyz != ixyz {
                                let ikxyz =
                                    ioff(max(ixyz, kxyz)) + min(ixyz, kxyz);
                                let jkxyz =
                                    ioff(max(jxyz, kxyz)) + min(jxyz, kxyz);

                                let mut val = 0.0;
                                for j in 0..nvib {
                                    let val1 = 1.0 / freq[j].powi(2)
                                        + 2.0 / freq[i].powi(2);
                                    let val2 = freq[j].powf(1.5)
                                        * (c[(j, jkxyz - 1)]
                                            * zmat[(j, i, ixyz - 1)]
                                            * rotcon[ixyz - 1]
                                            + c[(j, ikxyz - 1)]
                                                * zmat[(j, i, jxyz - 1)]
                                                * rotcon[jxyz - 1]
                                            + c[(j, ijxyz - 1)]
                                                * zmat[(j, i, kxyz - 1)]
                                                * rotcon[kxyz - 1]);
                                    val += val2 * val1 / freq[i].sqrt();
                                }
                                cc[(i, kxyz - 1, jxyz - 1, ixyz - 1)] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    cc
}

// went with the fortran indexing here and then subtracting one since that was
// easier than figuring out the normal indexing
fn t_mat(maxcor: usize, nvib: usize, freq: &Dvec, c: &Dmat) -> Dmat {
    let mut t = Dmat::zeros(maxcor, maxcor);
    for ixyz in 1..=maxcor {
        let iixyz = ioff(ixyz + 1);
        for jxyz in 1..=maxcor {
            let ijxyz = ioff(max(ixyz, jxyz)) + min(ixyz, jxyz);
            let jjxyz = ioff(jxyz + 1);
            let mut val = 0.0;
            for i in 0..nvib {
                val += freq[i] * c[(i, iixyz - 1)] * c[(i, jjxyz - 1)];
                if ixyz != jxyz {
                    val +=
                        2.0 * freq[i] * c[(i, ijxyz - 1)] * c[(i, ijxyz - 1)];
                }
            }
            t[(ixyz - 1, jxyz - 1)] = -0.5 * val;
        }
    }
    t
}

fn c_mat(
    maxcor: usize,
    nvib: usize,
    freq: &Dvec,
    primat: &Vec3,
    wila: &Dmat,
) -> Dmat {
    let mut c = Dmat::zeros(maxcor, 6);
    for i in 0..nvib {
        let cnst = (SQLAM * SQLAM * freq[i]).powf(1.5);
        for ixyz in 0..maxcor {
            for jxyz in 0..=ixyz {
                let ijxyz = ioff(ixyz + 1) + jxyz;
                let div = 2.0 * primat[ixyz] * primat[jxyz] * cnst;
                c[(i, ijxyz)] = wila[(i, ijxyz)] / div;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::dmatrix;

    use crate::{
        utils::{
            force3, force4, funds, load_fc2, load_fc3, load_fc4,
            symm_eigen_decomp, to_wavenumbers, xcalc,
        },
        CONST, FACT2,
    };

    use super::*;

    #[test]
    fn test_all() {
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
        let (xcnst, _e0) = xcalc(s.nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);
        let _fund = funds(&freq, s.nvib, &xcnst);

        let primat = s.geom.principal_moments();

        // c_mat
        let c = {
            let got = c_mat(3, 3, &freq, &primat, &wila);
            // sign of second element is swapped but that's okay since wila
            // comes from LXM
            let want = dmatrix![
                 0.00000000, -0.00070197, 0.00000000, 0.00000000, -0.00000000,
            0.00000000;
            -0.00098872, 0.00000000, -0.00053911, -0.00000000, 0.00000000,
            -0.00034891;
            -0.00488437, 0.00000000, 0.00136650,
                -0.00000000, 0.00000000, -0.00001189;
                           ];
            assert_abs_diff_eq!(got, want, epsilon = 1e-8);
            got
        };
        // t_mat
        {
            let got = t_mat(3, 3, &freq, &c);
            let want = dmatrix! {
            -0.02156707,  0.00254453, -0.00070920;
             0.00254453, -0.00209853, -0.00034716;
            -0.00070920, -0.00034716, -0.00023348;
            };
            assert_abs_diff_eq!(got, want, epsilon = 5e-8);
        }
    }
}
