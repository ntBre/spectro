use std::{fmt::Display, ops::Sub};

use approx::AbsDiffEq;

use crate::{
    utils::{make_tau, princ_cart, tau_prime},
    Dmat, Dvec, Spectro,
};

/// struct containing all of the quartic distortion constants and probably some
/// related values. delj, delk, deljk, sdelk, and sdelj are coefficients in the
/// Watson A Hamiltonian, while dj, djk, dk, and sd1 and the corresponding
/// values in the Watson S Hamiltonian. bxa, bya, and bza are the effective
/// rotational constants in Watson A, while bxs, bys, bzs are the corresponding
/// values in Watson S. djw, djkw, and dkw are Wilson's centrifugal distortion
/// constants. djn, djkn, dkn, sdjn, r5, and r6 are Nielsen distortion constants
#[allow(unused)]
#[cfg_attr(test, derive(serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Quartic {
    pub(crate) sigma: f64,
    pub(crate) rkappa: f64,
    pub(crate) delj: f64,
    pub(crate) delk: f64,
    pub(crate) deljk: f64,
    pub(crate) sdelk: f64,
    pub(crate) sdelj: f64,
    pub(crate) bxa: f64,
    pub(crate) bya: f64,
    pub(crate) bza: f64,
    pub(crate) djn: f64,
    pub(crate) djkn: f64,
    pub(crate) dkn: f64,
    pub(crate) sdjn: f64,
    pub(crate) r5: f64,
    pub(crate) r6: f64,
    pub(crate) dj: f64,
    pub(crate) djk: f64,
    pub(crate) dk: f64,
    pub(crate) sd1: f64,
    pub(crate) sd2: f64,
    pub(crate) bxs: f64,
    pub(crate) bys: f64,
    pub(crate) bzs: f64,
    pub(crate) djw: f64,
    pub(crate) djkw: f64,
    pub(crate) dkw: f64,
}

impl AbsDiffEq for Quartic {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.sigma.abs_diff_eq(&other.sigma, epsilon)
            && self.rkappa.abs_diff_eq(&other.rkappa, epsilon)
            && self.delj.abs_diff_eq(&other.delj, epsilon)
            && self.delk.abs_diff_eq(&other.delk, epsilon)
            && self.deljk.abs_diff_eq(&other.deljk, epsilon)
            && self.sdelk.abs_diff_eq(&other.sdelk, epsilon)
            && self.sdelj.abs_diff_eq(&other.sdelj, epsilon)
            && self.bxa.abs_diff_eq(&other.bxa, epsilon)
            && self.bya.abs_diff_eq(&other.bya, epsilon)
            && self.bza.abs_diff_eq(&other.bza, epsilon)
            && self.djn.abs_diff_eq(&other.djn, epsilon)
            && self.djkn.abs_diff_eq(&other.djkn, epsilon)
            && self.dkn.abs_diff_eq(&other.dkn, epsilon)
            && self.sdjn.abs_diff_eq(&other.sdjn, epsilon)
            && self.r5.abs_diff_eq(&other.r5, epsilon)
            && self.r6.abs_diff_eq(&other.r6, epsilon)
            && self.dj.abs_diff_eq(&other.dj, epsilon)
            && self.djk.abs_diff_eq(&other.djk, epsilon)
            && self.dk.abs_diff_eq(&other.dk, epsilon)
            && self.sd1.abs_diff_eq(&other.sd1, epsilon)
            && self.sd2.abs_diff_eq(&other.sd2, epsilon)
            && self.bxs.abs_diff_eq(&other.bxs, epsilon)
            && self.bys.abs_diff_eq(&other.bys, epsilon)
            && self.bzs.abs_diff_eq(&other.bzs, epsilon)
            && self.djw.abs_diff_eq(&other.djw, epsilon)
            && self.djkw.abs_diff_eq(&other.djkw, epsilon)
            && self.dkw.abs_diff_eq(&other.dkw, epsilon)
    }
}

impl Sub for Quartic {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            sigma: self.sigma - rhs.sigma,
            rkappa: self.rkappa - rhs.rkappa,
            delj: self.delj - rhs.delj,
            delk: self.delk - rhs.delk,
            deljk: self.deljk - rhs.deljk,
            sdelk: self.sdelk - rhs.sdelk,
            sdelj: self.sdelj - rhs.sdelj,
            bxa: self.bxa - rhs.bxa,
            bya: self.bya - rhs.bya,
            bza: self.bza - rhs.bza,
            djn: self.djn - rhs.djn,
            djkn: self.djkn - rhs.djkn,
            dkn: self.dkn - rhs.dkn,
            sdjn: self.sdjn - rhs.sdjn,
            r5: self.r5 - rhs.r5,
            r6: self.r6 - rhs.r6,
            dj: self.dj - rhs.dj,
            djk: self.djk - rhs.djk,
            dk: self.dk - rhs.dk,
            sd1: self.sd1 - rhs.sd1,
            sd2: self.sd2 - rhs.sd2,
            bxs: self.bxs - rhs.bxs,
            bys: self.bys - rhs.bys,
            bzs: self.bzs - rhs.bzs,
            djw: self.djw - rhs.djw,
            djkw: self.djkw - rhs.djkw,
            dkw: self.dkw - rhs.dkw,
        }
    }
}

impl Display for Quartic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "sigma: {:20.12}", self.sigma)?;
        writeln!(f, "rkappa: {:20.12}", self.rkappa)?;
        writeln!(f, "delj: {:20.12}", self.delj)?;
        writeln!(f, "delk: {:20.12}", self.delk)?;
        writeln!(f, "deljk: {:20.12}", self.deljk)?;
        writeln!(f, "sdelk: {:20.12}", self.sdelk)?;
        writeln!(f, "sdelj: {:20.12}", self.sdelj)?;
        writeln!(f, "bxa: {:20.12}", self.bxa)?;
        writeln!(f, "bya: {:20.12}", self.bya)?;
        writeln!(f, "bza: {:20.12}", self.bza)?;
        writeln!(f, "djn: {:20.12}", self.djn)?;
        writeln!(f, "djkn: {:20.12}", self.djkn)?;
        writeln!(f, "dkn: {:20.12}", self.dkn)?;
        writeln!(f, "sdjn: {:20.12}", self.sdjn)?;
        writeln!(f, "r5: {:20.12}", self.r5)?;
        writeln!(f, "r6: {:20.12}", self.r6)?;
        writeln!(f, "dj: {:20.12}", self.dj)?;
        writeln!(f, "djk: {:20.12}", self.djk)?;
        writeln!(f, "dk: {:20.12}", self.dk)?;
        writeln!(f, "sd1: {:20.12}", self.sd1)?;
        writeln!(f, "sd2: {:20.12}", self.sd2)?;
        writeln!(f, "bxs: {:20.12}", self.bxs)?;
        writeln!(f, "bys: {:20.12}", self.bys)?;
        writeln!(f, "bzs: {:20.12}", self.bzs)?;
        writeln!(f, "djw: {:20.12}", self.djw)?;
        writeln!(f, "djkw: {:20.12}", self.djkw)?;
        writeln!(f, "dkw: {:20.12}", self.dkw)
    }
}

impl Quartic {
    /// calculate the quartic centrifugal distortion constants
    #[allow(unused)]
    pub(crate) fn new(
        spectro: &Spectro,
        nvib: usize,
        freq: &Dvec,
        wila: &Dmat,
        rotcon: &[f64],
    ) -> Self {
        let maxcor = if spectro.is_linear() { 2 } else { 3 };
        let tau = make_tau(maxcor, nvib, freq, &spectro.primat, wila);
        let taupcm = tau_prime(maxcor, &tau);
        // NOTE: pretty sure this is always the case
        let irep = 0;
        let (ic, id) = princ_cart(irep);

        let mut t = Dmat::zeros(maxcor, maxcor);
        for ixyz in 0..maxcor {
            for jxyz in 0..maxcor {
                t[(ic[ixyz], ic[jxyz])] = taupcm[(ixyz, jxyz)] / 4.0;
            }
        }

        let t400 =
            (3.0e0 * t[(0, 0)] + 3.0e0 * t[(1, 1)] + 2.0e0 * t[(0, 1)]) / 8.0e0;
        let t220 = t[(0, 2)] + t[(1, 2)] - 2.0e0 * t400;
        let t040 = t[(2, 2)] - t220 - t400;
        let t202 = (t[(0, 0)] - t[(1, 1)]) / 4.0e0;
        let t022 = (t[(0, 2)] - t[(1, 2)]) / 2.0e0 - t202;
        let t004 = (t[(0, 0)] + t[(1, 1)] - 2.0e0 * t[(0, 1)]) / 16.0e0;

        let b200 = 0.5 * (rotcon[id[0]] + rotcon[id[1]]) - 4.0 * t004;
        let b020 = rotcon[id[2]] - b200 + 6.0 * t004;
        let b002 = 0.25 * (rotcon[id[0]] - rotcon[id[1]]);

        let sigma = (2.0 * rotcon[id[2]] - rotcon[id[0]] - rotcon[id[1]])
            / (rotcon[id[0]] - rotcon[id[1]]);
        let djw = -taupcm[(ic[0], ic[0])] / 4.0;
        Quartic {
            // asymmetric top
            sigma: (2.0 * rotcon[id[2]] - rotcon[id[0]] - rotcon[id[1]])
                / (rotcon[id[0]] - rotcon[id[1]]),
            // definitely need not to do this if it's not an asymmetric top
            rkappa: (2.0 * rotcon[1] - rotcon[0] - rotcon[2])
                / (rotcon[0] - rotcon[2]),
            // coefficients in the Watson A reduction
            delj: -t400 - 2.0 * t004,
            delk: -t040 - 10.0 * t004,
            deljk: -t220 + 12.0 * t004,
            sdelk: -t022 - 4.0 * sigma * t004,
            sdelj: -t202,
            // effective rotational constants
            bxa: rotcon[id[0]] - 8.0 * (sigma + 1.0) * t004,
            bya: rotcon[id[1]] + 8.0 * (sigma - 1.0) * t004,
            bza: rotcon[id[2]] + 16.0 * t004,
            // nielsen centrifugal distortion constants
            djn: -t400,
            djkn: -t220,
            dkn: -t040,
            sdjn: -t202,
            r5: t022 / 2.0,
            r6: t004,
            // coefficients in the Watson S reduction
            dj: -t400 + 0.5 * t022 / sigma,
            djk: -t220 - 3.0 * t022 / sigma,
            dk: -t040 + 2.5 * t022 / sigma,
            sd1: t202,
            sd2: t004 + 0.25 * t022 / sigma,
            // effective rotational constants again
            bxs: rotcon[id[0]] - 4.0 * t004 + (2.0 + 1.0 / sigma) * t022,
            bys: rotcon[id[1]] - 4.0 * t004 - (2.0 - 1.0 / sigma) * t022,
            bzs: rotcon[id[2]] + 6.0 * t004 - 5.0 * t022 / (2.0 * sigma),
            // Wilson's centrifugal distortion constants
            djw: -taupcm[(ic[0], ic[0])] / 4.0,
            djkw: -2.0 * djw - taupcm[(ic[0], ic[2])] / 2.0,
            dkw: djw - taupcm[(ic[2], ic[2])] / 4.0
                + taupcm[(ic[0], ic[2])] / 2.0,
        }
    }
}
