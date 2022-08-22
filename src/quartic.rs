use crate::{
    utils::{make_tau, princ_cart, tau_prime},
    Dmat, Dvec, Spectro,
};

/// struct containing all of the quartic distortion constants and probably some
/// related values
#[allow(unused)]
pub(crate) struct Quartic {
    pub(crate) sigma: f64,
    pub(crate) rkappa: f64,

    // coefficients in the Watson A reduction
    pub(crate) delj: f64,
    pub(crate) delk: f64,
    pub(crate) deljk: f64,
    pub(crate) sdelk: f64,
    pub(crate) sdelj: f64,

    // effective rotational constants
    pub(crate) bxa: f64,
    pub(crate) bya: f64,
    pub(crate) bza: f64,

    // nielsen centrifugal distortion constants
    pub(crate) djn: f64,
    pub(crate) djkn: f64,
    pub(crate) dkn: f64,
    pub(crate) sdjn: f64,
    pub(crate) r5: f64,
    pub(crate) r6: f64,

    // coefficients in the Watson S reduction
    pub(crate) dj: f64,
    pub(crate) djk: f64,
    pub(crate) dk: f64,
    pub(crate) sd1: f64,
    pub(crate) sd2: f64,

    pub(crate) bxs: f64,
    pub(crate) bys: f64,
    pub(crate) bzs: f64,

    // Wilson's centrifugal distortion constants
    pub(crate) djw: f64,
    pub(crate) djkw: f64,
    pub(crate) dkw: f64,
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
        let primat = spectro.geom.principal_moments();
        let tau = make_tau(maxcor, nvib, freq, &primat, wila);
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
            djkw: -2.0 * djw - taupcm[(ic[0], ic[1])] / 2.0,
            dkw: djw - taupcm[(ic[2], ic[2])] / 4.0
                + taupcm[(ic[0], ic[2])] / 2.0,
        }
    }
}
