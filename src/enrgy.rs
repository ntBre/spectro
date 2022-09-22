use super::{Dmat, Dvec, Spectro};
use crate::{
    f3qcm::F3qcm, ioff, resonance::Restst, rsfrm1, rsfrm2, state::State, Mode,
};
use std::cmp::{max, min};

impl Spectro {
    /// vibrational energy levels and properties in resonance. returns the
    /// energies in the same order as the states in `i1sts`
    pub(crate) fn enrgy(
        &self,
        freq: &Dvec,
        xcnst: &Dmat,
        gcnst: &Option<Dmat>,
        restst: &Restst,
        f3qcm: &F3qcm,
        e0: f64,
        eng: &mut [f64],
    ) {
        let Restst {
            coriolis: _,
            fermi1,
            fermi2,
            darling: _,
            states,
            modes,
            ifunda,
            iovrtn,
            icombn,
        }: &Restst = restst;

        let nstate = states.len();
        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let (i1sts, i2sts, _) = State::partition(states);

        part1(
            nstate, &i1mode, freq, i1sts, &i2mode, i2sts, n1dm, xcnst, n2dm,
            gcnst, eng, e0,
        );

        let (_, ifrm1, ifrm2) = self.make_fermi_checks(fermi1, fermi2);

        if self.rotor.is_sym_top() {
            // NOTE I think this is not going to work at all :( I think my
            // ist/jst stuff inside rsfrm1 is going to break spectacularly here
            // but we'll see
            for ii in 0..n1dm {
                let ivib = i1mode[ii];
                // type 1 fermi resonance
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, false);
                }

                // type 2 fermi resonance
                for jj in ii + 1..n1dm {
                    let jvib = i1mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        // +1 because that's how I inserted them in restst
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }

                for jj in 0..n2dm {
                    let (jvib, _) = i2mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }

            // type 1 again
            for ii in 0..n2dm {
                let (ivib, _) = i2mode[ii];
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, true);
                }

                // type 2 again
                for jj in ii + 1..n2dm {
                    let (jvib, _) = i2mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }
        } else {
            for iii in 0..n1dm {
                let ivib = i1mode[iii];
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, false);
                }
                for jjj in iii + 1..n1dm {
                    let jvib = i1mode[jjj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }
        }
    }
}

/// first part of the [enrgy] computation
pub(crate) fn part1(
    nstate: usize,
    i1mode: &[usize],
    freq: &Dvec,
    i1sts: Vec<Vec<usize>>,
    i2mode: &[(usize, usize)],
    i2sts: Vec<Vec<(usize, usize)>>,
    n1dm: usize,
    xcnst: &Dmat,
    n2dm: usize,
    gcnst: &Option<Dmat>,
    eng: &mut [f64],
    e0: f64,
) {
    for nst in 0..nstate {
        let mut val1 = 0.0;
        for (ii, &i) in i1mode.iter().enumerate() {
            val1 += freq[i] * ((i1sts[nst][ii] as f64) + 0.5);
        }

        let mut val2 = 0.0;
        for (ii, &(i, _)) in i2mode.iter().enumerate() {
            val2 += freq[i] * (i2sts[nst][ii].0 as f64 + 1.0);
        }

        // this is val2 in the asym top code
        let mut val3 = 0.0;
        for ii in 0..n1dm {
            let i = i1mode[ii];
            for jj in 0..=ii {
                let j = i1mode[jj];
                val3 += xcnst[(i, j)]
                    * ((i1sts[nst][ii] as f64) + 0.5)
                    * ((i1sts[nst][jj] as f64) + 0.5);
            }
        }

        let mut val4 = 0.0;
        for (ii, &i) in i1mode.iter().enumerate() {
            for (jj, &(j, _)) in i2mode.iter().enumerate() {
                val4 += xcnst[(i, j)]
                    * ((i1sts[nst][ii] as f64 + 0.5)
                        * (i2sts[nst][jj].0 as f64 + 1.0));
            }
        }

        let mut val5 = 0.0;
        for ii in 0..n2dm {
            let i = i2mode[ii].0;
            for jj in 0..=ii {
                let j = i2mode[jj].0;
                val5 += xcnst[(i, j)]
                    * ((i2sts[nst][ii].0 as f64 + 1.0)
                        * (i2sts[nst][jj].0 as f64 + 1.0));
            }
        }

        let mut val6 = 0.0;
        for (ii, &(i, _)) in i2mode.iter().enumerate() {
            for (jj, &(j, _)) in i2mode.iter().take(ii + 1).enumerate() {
                val6 += gcnst
                    .as_ref()
                    .expect("g constants required for symmetric tops")[(i, j)]
                    * (i2sts[nst][ii].1 as f64)
                    * (i2sts[nst][jj].1 as f64);
            }
        }

        eng[nst] = val1 + val2 + val3 + val4 + val5 + val6 + e0;
    }
}
