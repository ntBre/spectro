use std::{
    cmp::{max, min},
    fmt::Display,
};

use approx::AbsDiffEq;
use tensor::Tensor4;
type Tensor3 = tensor::tensor3::Tensor3<f64>;

use crate::{
    consts::SQLAM,
    f3qcm::F3qcm,
    utils::{ioff, make_tau, princ_cart, tau_prime},
    Dmat, Dvec, Spectro,
};

/// struct holding the sextic distortion constants. For an asymmetric top, the
/// field names are correct. For a symmetric top, phi->H and sphij->h1,
/// sphijk->h2, and sphik->h3. TODO make this an enum to get rid of this comment
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sextic {
    pub(crate) phij: f64,
    pub(crate) phijk: f64,
    pub(crate) phikj: f64,
    pub(crate) phik: f64,
    pub(crate) sphij: f64,
    pub(crate) sphijk: f64,
    pub(crate) sphik: f64,
}

impl AbsDiffEq for Sextic {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.phij.abs_diff_eq(&other.phij, epsilon)
            && self.phijk.abs_diff_eq(&other.phijk, epsilon)
            && self.phikj.abs_diff_eq(&other.phikj, epsilon)
            && self.phik.abs_diff_eq(&other.phik, epsilon)
            && self.sphij.abs_diff_eq(&other.sphij, epsilon)
            && self.sphijk.abs_diff_eq(&other.sphijk, epsilon)
            && self.sphik.abs_diff_eq(&other.sphik, epsilon)
    }
}

fn scc(
    maxcor: usize,
    tau: Tensor4,
    rotcon: &[f64],
    nvib: usize,
    freq: &Dvec,
    cc: Tensor4,
    f3qcm: &F3qcm,
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
                    val3 += f3qcm[(i, j, k)]
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
                            valc += f3qcm[(i, j, k)] * c[(k, iixyz - 1)] * val1;
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
                    if kxyz != jxyz && kxyz != ixyz {
                        let div =
                            4.0 * (rotcon[ixyz - 1] - rotcon[kxyz - 1]).powi(2);
                        if div > TOL {
                            let val1 = (rotcon[(ixyz - 1)]
                                - rotcon[(kxyz - 1)])
                                * (tau
                                    [(kxyz - 1, ixyz - 1, jxyz - 1, jxyz - 1)]
                                    + 2.0e0
                                        * tau[(
                                            kxyz - 1,
                                            jxyz - 1,
                                            jxyz - 1,
                                            ixyz - 1,
                                        )]);
                            let val2 = (rotcon[(ixyz - 1)]
                                - rotcon[(jxyz - 1)])
                                * (tau
                                    [(kxyz - 1, kxyz - 1, kxyz - 1, ixyz - 1)]
                                    - tau[(
                                        kxyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                        ixyz - 1,
                                    )]);
                            vale += tau
                                [(kxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)]
                                * (val1 + val2)
                                / div;
                        }

                        let div =
                            8.0 * (rotcon[jxyz - 1] - rotcon[kxyz - 1]).powi(2);
                        if div > TOL {
                            let val1 = (rotcon[(jxyz - 1)]
                                - rotcon[(kxyz - 1)])
                                * (tau
                                    [(kxyz - 1, jxyz - 1, ixyz - 1, ixyz - 1)]
                                    + 2.0
                                        * tau[(
                                            kxyz - 1,
                                            ixyz - 1,
                                            jxyz - 1,
                                            ixyz - 1,
                                        )]);
                            let val2 = 2.0
                                * (rotcon[(ixyz - 1)] - rotcon[(jxyz - 1)])
                                * (tau
                                    [(kxyz - 1, jxyz - 1, jxyz - 1, jxyz - 1)]
                                    - tau[(
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
                            valf += val3 * (val1 + val2) / div;
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
        vala += (val1 + val2 + val3 + val4) / rotcon[(ixyz - 1)];
    }
    vala = 3.0 * vala / 16.0;
    let mut valb = 0.0;
    for i in 0..nvib {
        let val1 = 2.0 * cc[(i, 2, 1, 0)].powi(2)
            + cc[(i, 1, 1, 0)] * cc[(i, 2, 2, 0)]
            + cc[(i, 2, 2, 1)] * cc[(i, 1, 0, 0)]
            + cc[(i, 2, 0, 0)] * cc[(i, 2, 1, 1)];
        valb += freq[(i)] * val1;
    }
    valb *= 2.0;
    let mut valc = 0.0;
    for i in 0..nvib {
        for j in 0..nvib {
            for k in 0..nvib {
                let val1 = c[(i, 0)] * c[(j, 2)] * c[(k, 5)]
                    + 2.0 * c[(i, 0)] * c[(j, 4)] * c[(k, 4)]
                    + 2.0 * c[(i, 2)] * c[(j, 3)] * c[(k, 3)]
                    + 2.0 * c[(i, 5)] * c[(j, 1)] * c[(k, 1)]
                    + 8.0 * c[(i, 4)] * c[(j, 3)] * c[(k, 1)];
                valc += f3qcm[(i, j, k)] * val1;
            }
        }
    }
    valc *= 0.5e0;
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
                                    * zmat[(j, i, ixyz - 1)]
                                    * rotcon[ixyz - 1]);
                        val += val2 * val1 / freq[i].sqrt();
                    }
                    cc[(i, jxyz - 1, ixyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, jxyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, ixyz - 1, jxyz - 1)] = val;

                    for kxyz in 1..=maxcor {
                        if kxyz != jxyz && kxyz != ixyz {
                            let ikxyz = ioff(max(ixyz, kxyz)) + min(ixyz, kxyz);
                            let jkxyz = ioff(max(jxyz, kxyz)) + min(jxyz, kxyz);

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
    cc
}

pub(crate) fn c_mat(
    maxcor: usize,
    nvib: usize,
    freq: &Dvec,
    primat: &[f64],
    wila: &Dmat,
) -> Dmat {
    let mut c = Dmat::zeros(nvib, 6);
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

impl Sextic {
    pub fn new(
        s: &Spectro,
        wila: &Dmat,
        zmat: &Tensor3,
        freq: &Dvec,
        f3qcm: &F3qcm,
    ) -> Self {
        let mut ret = Self::default();
        // the only thing we can compute for linear molecules is He, which is
        // phi(1, 1, 1) in fortran
        if s.rotor.is_linear() {
            return ret;
        }
        let nvib = s.nvib;
        let maxcor = if s.is_linear() { 2 } else { 3 };
        let c = c_mat(maxcor, nvib, freq, &s.primat, wila);
        let cc = cc_tensor(nvib, maxcor, freq, &c, zmat, &s.rotcon);
        let tau = make_tau(maxcor, nvib, freq, &s.primat, wila);
        let taucpm = tau_prime(maxcor, &tau);
        let scc = scc(maxcor, tau, &s.rotcon, nvib, freq, cc, f3qcm, &c, s);
        // says this is the default representation and sets it to 1 if it was
        // originally 0. NOTE my 0 is fortran 1
        let irep = if s.rotor.is_sym_top() { 5 } else { 0 };
        let (ic, id) = princ_cart(irep);
        let mut t = Dmat::zeros(maxcor, maxcor);
        for ixyz in 0..maxcor {
            for jxyz in 0..maxcor {
                t[(ic[ixyz], ic[jxyz])] = taucpm[(ixyz, jxyz)] / 4.0;
            }
        }
        let phi = make_phi(maxcor, ic, scc);
        let t400 =
            (3.0 * t[(0, 0)] + 3.0 * t[(2 - 1, 2 - 1)] + 2.0 * t[(0, 2 - 1)])
                / 8.0;
        let t220 = t[(0, 3 - 1)] + t[(2 - 1, 3 - 1)] - 2.0 * t400;
        let t040 = t[(3 - 1, 3 - 1)] - t220 - t400;
        let t202 = (t[(0, 0)] - t[(2 - 1, 2 - 1)]) / 4.0;
        let t022 = (t[(0, 3 - 1)] - t[(2 - 1, 3 - 1)]) / 2.0 - t202;
        let t004 = (t[(0, 0)] + t[(2 - 1, 2 - 1)] - 2.0 * t[(0, 2 - 1)]) / 16.0;
        let b002 = 0.25e0 * (s.rotcon[id[0]] - s.rotcon[id[2 - 1]]);
        let phi600 = 5.0 * (phi[(0, 0, 0)] + phi[(2 - 1, 2 - 1, 2 - 1)]) / 16.0
            + (phi[(0, 0, 2 - 1)] + phi[(2 - 1, 2 - 1, 0)]) / 8.0;
        let phi420 = 3.0 * (phi[(0, 0, 3 - 1)] + phi[(2 - 1, 2 - 1, 3 - 1)])
            / 4.0
            + phi[(0, 2 - 1, 3 - 1)] / 4.0
            - 3.0 * phi600;
        let phi240 = phi[(3 - 1, 3 - 1, 0)] + phi[(3 - 1, 3 - 1, 2 - 1)]
            - 2.0 * phi420
            - 3.0 * phi600;
        let phi060 = phi[(3 - 1, 3 - 1, 3 - 1)] - phi240 - phi420 - phi600;
        let phi402 = 15.0 * (phi[(0, 0, 0)] - phi[(1, 1, 1)]) / 64.0
            + (phi[(0, 0, 1)] - phi[(1, 1, 0)]) / 32.0;
        let phi222 = (phi[(0, 0, 2)] - phi[(1, 1, 2)]) / 2.0 - 2.0 * phi402;
        let phi042 = (phi[(2, 2, 0)] - phi[(2, 2, 1)]) / 2.0 - phi222 - phi402;
        let phi204 = 3.0 * (phi[(0, 0, 0)] + phi[(2 - 1, 2 - 1, 2 - 1)]) / 32.0
            - (phi[(0, 0, 2 - 1)] + phi[(2 - 1, 2 - 1, 0)]) / 16.0;
        let phi024 = (phi[(0, 0, 3 - 1)] + phi[(2 - 1, 2 - 1, 3 - 1)]
            - phi[(0, 2 - 1, 3 - 1)])
            / 8.0
            - phi204;
        let phi006 = (phi[(0, 0, 0)] - phi[(2 - 1, 2 - 1, 2 - 1)]) / 64.0
            - (phi[(0, 0, 2 - 1)] - phi[(2 - 1, 2 - 1, 0)]) / 32.0;

        if !s.rotor.is_sym_top() {
            // asymmetric top
            let sigma = (2.0 * s.rotcon[id[3 - 1]]
                - s.rotcon[id[0]]
                - s.rotcon[id[2 - 1]])
                / (s.rotcon[id[0]] - s.rotcon[id[2 - 1]]);
            ret.phij = phi600 + 2.0 * phi204;
            ret.phijk = phi420 - 12.0 * phi204
                + 2.0 * phi024
                + 16.0 * sigma * phi006
                + 8.0 * t022 * t004 / b002;
            ret.phikj = phi240 + 10.0 * phi420 / 3.0
                - 30.0 * phi204
                - 10.0 * ret.phijk / 3.0;
            ret.phik = phi060 - 7.0 * phi420 / 3.0
                + 28.0 * phi204
                + 7.0 * ret.phijk / 3.0;
            ret.sphij = phi402 + phi006;
            ret.sphijk = phi222 + 4.0 * sigma * phi204 - 10.0 * phi006
                + 2.0 * (t220 - 2.0 * sigma * t202 - 4.0 * t004) * t004 / b002;
            ret.sphik = phi042
                + 4.0 * sigma * phi024 / 3.0
                + (32.0 * sigma * sigma / 3.0 + 9.0) * phi006
                + 4.0
                    * (t040 + sigma * t022 / 3.0
                        - 2.0 * (sigma * sigma - 2.0) * t004)
                    * t004
                    / b002;
        } else {
            let irep = if s.rotor.is_prolate() { 0 } else { 5 };
            let (_, id) = princ_cart(irep);
            let rkappa =
                (2.0 * s.rotcon[id[0]] - s.rotcon[id[1]] - s.rotcon[id[2]])
                    / (s.rotcon[id[1]] - s.rotcon[id[2]]);
            // they call bp bo for oblate, but I'll use bp for both
            let sigma = if rkappa < 0.0 {
                -999999999999.0
            } else {
                let bo = (rkappa - 1.0) / (rkappa + 3.0);
                1.0 / bo
            };

            let b200 = 0.3 * (s.rotcon[id[0]] + s.rotcon[id[1]]) - 4.0 * t004;
            let b020 = s.rotcon[id[2]] - b200 + 6.0 * t004;

            let div = 2.0 * sigma * sigma + 27.0 / 16.0;
            let mu = (sigma * phi042 - 9.0 * phi024 / 8.0
                + (-2.0 * sigma * t040 + (sigma * sigma + 3.0) * t022
                    - 5.0 * sigma * t004)
                    * t022
                    / b020)
                / div;
            let nu = 3.0 * mu / (16.0 * sigma)
                + phi024 / (8.0 * sigma)
                + t004 * t022 / b020;
            let lamda = 5.0 * nu / sigma
                + phi222 / (sigma * 2.0)
                + (-t220 / (sigma * 2.0) + t202
                    - t022 / (sigma * sigma)
                    - 2.0 * t004 / sigma)
                    * t022
                    / b020;

            ret.phij = phi600 - lamda;
            ret.phijk = phi420 + 6.0 * lamda - 3.0 * mu;
            ret.phikj = phi240 - 5.0 * lamda + 10.0 * mu;
            ret.phik = phi060 - 7.0 * mu;
            ret.sphij = phi402 - nu;
            ret.sphijk = phi204 + lamda / 2.0;
            ret.sphik = phi006 + nu;
        };

        // TODO linear molecule case and spherical top case

        ret
    }
}

fn make_phi(
    maxcor: usize,
    ic: [usize; 3],
    scc: tensor::Tensor3<f64>,
) -> tensor::Tensor3<f64> {
    let mut phi = Tensor3::zeros(maxcor, maxcor, maxcor);
    for ixyz in 0..maxcor {
        for jxyz in 0..maxcor {
            for kxyz in 0..maxcor {
                phi[(ic[ixyz], ic[jxyz], ic[kxyz])] = scc[(ixyz, jxyz, kxyz)];
            }
        }
    }
    phi
}

impl Display for Sextic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Watson A Reduction (S for symmetric tops)")?;
        writeln!(f, "Phi  J: {:20.12e}", self.phij)?;
        writeln!(f, "Phi  K: {:20.12e}", self.phik)?;
        writeln!(f, "Phi JK: {:20.12e}", self.phijk)?;
        writeln!(f, "Phi KJ: {:20.12e}", self.phikj)?;
        writeln!(f, "phi  J: {:20.12e}", self.sphij)?;
        writeln!(f, "phi JK: {:20.12e}", self.sphijk)?;
        writeln!(f, "phi  k: {:20.12e}", self.sphik)?;

        Ok(())
    }
}
