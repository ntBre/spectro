#![allow(clippy::if_same_then_else)]

use crate::f3qcm::F3qcm;
use crate::resonance::{Coriolis, Darling, Fermi1, Fermi2};
use crate::{mode::Mode, Dvec};
use std::fmt::Display;
type Tensor3 = tensor::tensor3::Tensor3<f64>;

// TODO probably need to get rid of default here. it will complicate the spectro
// usage a little bit but then I don't have to have the None variant and keep
// checking for it
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Rotor {
    Diatomic,
    Linear,
    SphericalTop,
    OblateSymmTop,
    ProlateSymmTop,
    AsymmTop,
    #[default]
    None,
}

#[macro_export]
macro_rules! unset_rotor {
    () => {
        panic!("rotor type not set")
    };
}

/// both Fermi resonance tolerance for f3qcm
const F3TOL: f64 = 10.0;

/// type 1 Fermi resonance tolerances
const FTOL1: f64 = 200.0;

/// type 2 Fermi resonance tolerances
const DLTOL: f64 = 1000.0;
const DFTOL: f64 = 200.0;

/// darling-dennison resonance tolerance
const DDTOL: f64 = 300.0;

impl Rotor {
    pub fn coriolis(
        &self,
        modes: &[Mode],
        freq: &Dvec,
        zmat: &Tensor3,
    ) -> Vec<Coriolis> {
        use Rotor::*;
        // tolerances for resonance checking
        const CTOL: f64 = 200.0;
        const ZTOL: f64 = 0.25;

        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let mut ret = Vec::new();
        match self {
            Diatomic => todo!(),
            // I think linear and spherical tops are handled with the other symm
            // tops, but make myself think about it again when I cross that
            // bridge
            SphericalTop => todo!(),
            OblateSymmTop | ProlateSymmTop | Linear => {
                let ia = 2;
                let ib = 1;
                let ic = 0;
                // not entirely sure about this, but I needed to swap a and b at
                // least for bipy
                let (a, b, c) = if self.is_prolate() {
                    (0, 2, 1)
                } else {
                    (ia, ib, ic)
                };
                for ii in 0..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..n1dm {
                        let j = i1mode[jj];
                        if ii == jj {
                            continue;
                        }
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL && zmat[(i, j, ia)].abs() >= ZTOL
                        {
                            ret.push(Coriolis::new(i, j, a));
                        }
                    }

                    for jj in 0..n2dm {
                        let j = i2mode[jj].0;
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL {
                            if zmat[(i, j, ib)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, b));
                            }
                            if zmat[(i, j, ic)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, c));
                            }
                        }
                    }
                } // end loop 310

                for ii in 0..n2dm {
                    let i = i2mode[ii].0;
                    for jj in 0..n2dm {
                        if ii == jj {
                            continue;
                        }
                        let j = i2mode[jj].1;
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL && zmat[(i, j, ia)].abs() >= ZTOL
                        {
                            ret.push(Coriolis::new(i, j, a));
                        }
                    }

                    for jj in 0..n1dm {
                        let j = i1mode[jj];
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL {
                            if zmat[(i, j, ib)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, b));
                            }
                            if zmat[(i, j, ic)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, c));
                            }
                        }
                    }

                    for jj in 0..n2dm {
                        if ii == jj {
                            continue;
                        }
                        let j = i2mode[jj].0;
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL {
                            if zmat[(i, j, ib)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, b));
                            }
                            if zmat[(i, j, ic)].abs() >= ZTOL {
                                ret.push(Coriolis::new(i, j, c));
                            }
                        }
                    }
                } // end 350 loop
            }
            AsymmTop => {
                for ii in 0..n1dm {
                    // loop over i1mode?
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        let diff = freq[i] - freq[j];
                        if diff.abs() <= CTOL {
                            for z in 0..3 {
                                if zmat[(i, j, z)].abs() >= ZTOL {
                                    ret.push(Coriolis::new(i, j, z));
                                }
                            }
                        }
                    }
                }
            }
            Rotor::None => unset_rotor!(),
        }
        ret
    }

    pub fn darling(&self, modes: &[Mode], freq: &Dvec) -> Vec<Darling> {
        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let mut ret = Vec::new();
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop | Rotor::ProlateSymmTop | Rotor::Linear => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        darling_test(freq, i, j, &mut ret);
                    }

                    for jj in 0..n2dm {
                        let j = i2mode[jj].0;
                        darling_test(freq, i, j, &mut ret);
                    }
                }

                for ii in 1..n2dm {
                    let i = i2mode[ii].0;
                    for jj in 0..ii {
                        let j = i2mode[jj].0;
                        darling_test(freq, i, j, &mut ret);
                    }
                }
            }
            Rotor::AsymmTop => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        darling_test(freq, i, j, &mut ret);
                    }
                }
            }
            Rotor::None => unset_rotor!(),
        }
        ret
    }

    pub fn fermi1(
        &self,
        modes: &[Mode],
        freq: &Dvec,
        f3qcm: &F3qcm,
    ) -> Vec<Fermi1> {
        let mut ret = Vec::new();
        use Rotor::*;
        let (i1mode, i2mode, _) = Mode::partition(modes);
        match self {
            Diatomic => todo!(),
            SphericalTop => todo!(),
            OblateSymmTop | ProlateSymmTop | Linear => {
                for i in &i1mode {
                    for j in &i1mode {
                        ferm1_test(freq, *i, *j, f3qcm, &mut ret);
                    }
                }

                for (i, _) in &i2mode {
                    for j in &i1mode {
                        ferm1_test(freq, *i, *j, f3qcm, &mut ret);
                    }

                    for (j, _) in &i2mode {
                        ferm1_test(freq, *i, *j, f3qcm, &mut ret);
                    }
                }
            }
            AsymmTop => {
                for i in &i1mode {
                    for j in &i1mode {
                        ferm1_test(freq, *i, *j, f3qcm, &mut ret);
                    }
                }
            }
            Rotor::None => unset_rotor!(),
        }
        ret
    }

    pub fn fermi2(
        &self,
        modes: &[Mode],
        freq: &Dvec,
        f3qcm: &F3qcm,
    ) -> Vec<Fermi2> {
        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let mut ret = Vec::new();
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop | Rotor::ProlateSymmTop | Rotor::Linear => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];

                    for jj in 0..ii {
                        let j = i1mode[jj];
                        for kk in 0..n1dm {
                            let k = i1mode[kk];
                            if ferm2_test(freq, i, j, k, f3qcm, &mut ret) {
                                continue;
                            }
                        }
                    }

                    for jj in 0..n2dm {
                        let j = i2mode[jj].0;
                        for kk in 0..n2dm {
                            let k = i2mode[kk].0;
                            if ferm2_test(freq, i, j, k, f3qcm, &mut ret) {
                                continue;
                            }
                        }
                    }
                }

                for ii in 1..n2dm {
                    let i = i2mode[ii].0;

                    for jj in 0..n2dm - 1 {
                        let j = i2mode[jj].0;
                        for kk in 0..n1dm {
                            let k = i1mode[kk];
                            if ferm2_test(freq, i, j, k, f3qcm, &mut ret) {
                                continue;
                            }
                        }

                        for kk in 0..n2dm {
                            let k = i2mode[kk].0;
                            if ferm2_test(freq, i, j, k, f3qcm, &mut ret) {
                                continue;
                            }
                        }
                    }
                }

                // jan martin can't believe this section, "can someone please
                // explain me the point of this?(!)"
                for ii in 0..n2dm {
                    let i = i2mode[ii].0;

                    for jj in 0..ii {
                        let j = i2mode[jj].0;
                        for kk in 0..n1dm {
                            let k = i1mode[kk];
                            let diff1 = freq[i] - freq[j] - freq[k];
                            let diff2 = -freq[i] + freq[j] - freq[k];
                            let diff3 = -freq[i] - freq[j] + freq[k];
                            let diff4 = freq[i] + freq[j] + freq[k];
                            let delta = diff1 * diff2 * diff3 * diff4;
                            let ijk = (i, j, k);
                            // if we just reverse the order of the condition
                            // checks, we can only check f3qcm once...
                            if delta.abs() <= DLTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff1.abs() <= DFTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff2.abs() <= DFTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff3.abs() <= DFTOL
                                && f3qcm[ijk].abs() >= F3TOL
                            {
                                ret.push(Fermi2::new(i, j, k));
                                continue;
                            }
                        }

                        for kk in 0..n2dm {
                            let k = i2mode[kk].0;
                            // copy-pasted from above
                            let diff1 = freq[i] - freq[j] - freq[k];
                            let diff2 = -freq[i] + freq[j] - freq[k];
                            let diff3 = -freq[i] - freq[j] + freq[k];
                            let diff4 = freq[i] + freq[j] + freq[k];
                            let delta = diff1 * diff2 * diff3 * diff4;
                            let ijk = (i, j, k);
                            // if we just reverse the order of the condition
                            // checks, we can only check f3qcm once...
                            if delta.abs() <= DLTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff1.abs() <= DFTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff2.abs() <= DFTOL {
                                if f3qcm[ijk].abs() >= F3TOL {
                                    ret.push(Fermi2::new(i, j, k));
                                    continue;
                                }
                            } else if diff3.abs() <= DFTOL
                                && f3qcm[ijk].abs() >= F3TOL
                            {
                                ret.push(Fermi2::new(i, j, k));
                                continue;
                            }
                        }
                    }
                }
            }
            Rotor::AsymmTop => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        for kk in 0..jj {
                            let k = i1mode[kk];
                            if i != j
                                && j != k
                                && i != k
                                && ferm2_test(freq, i, j, k, f3qcm, &mut ret)
                            {
                                continue;
                            }
                        }
                    }
                }
            }
            Rotor::None => unset_rotor!(),
        }
        ret
    }

    /// panics if `self` is not set
    pub fn is_prolate(&self) -> bool {
        assert!(*self != Rotor::None);
        *self == Rotor::ProlateSymmTop
    }

    /// Report whether or not `self` is either an `OblateSymmTop` or a
    /// `ProlateSymmTop`. panics if `self` is not set
    pub fn is_sym_top(&self) -> bool {
        use Rotor::*;
        match &self {
            Diatomic => false,
            SphericalTop => false,
            OblateSymmTop | ProlateSymmTop | Linear => true,
            AsymmTop => false,
            None => unset_rotor!(),
        }
    }

    /// Returns `true` if the rotor is [`Linear`].
    ///
    /// [`Linear`]: Rotor::Linear
    #[must_use]
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear)
    }

    /// Returns `true` if the rotor is [`OblateSymmTop`].
    ///
    /// [`OblateSymmTop`]: Rotor::OblateSymmTop
    #[must_use]
    pub fn is_oblate(&self) -> bool {
        matches!(self, Self::OblateSymmTop)
    }

    /// Returns `true` if the rotor is [`ProlateSymmTop`].
    ///
    /// [`ProlateSymmTop`]: Rotor::ProlateSymmTop
    #[must_use]
    pub fn is_prolate_symm_top(&self) -> bool {
        matches!(self, Self::ProlateSymmTop)
    }

    /// Returns `true` if the rotor is [`Diatomic`].
    ///
    /// [`Diatomic`]: Rotor::Diatomic
    #[must_use]
    pub fn is_diatomic(&self) -> bool {
        matches!(self, Self::Diatomic)
    }

    /// Returns `true` if the rotor is [`SphericalTop`].
    ///
    /// [`SphericalTop`]: Rotor::SphericalTop
    #[must_use]
    pub fn is_spherical_top(&self) -> bool {
        matches!(self, Self::SphericalTop)
    }

    /// Returns `true` if the rotor is [`AsymmTop`].
    ///
    /// [`AsymmTop`]: Rotor::AsymmTop
    #[must_use]
    pub fn is_asymm_top(&self) -> bool {
        matches!(self, Self::AsymmTop)
    }
}

fn ferm1_test(
    freq: &Dvec,
    i: usize,
    j: usize,
    f3qcm: &F3qcm,
    ret: &mut Vec<Fermi1>,
) {
    let diff = 2.0 * freq[i] - freq[j];
    if diff.abs() <= FTOL1 && f3qcm[(i, i, j)].abs() >= F3TOL {
        ret.push(Fermi1::new(i, j));
    }
}

fn darling_test(freq: &Dvec, i: usize, j: usize, ret: &mut Vec<Darling>) {
    let diff = 2.0 * freq[i] - 2.0 * freq[j];
    if diff.abs() <= DDTOL {
        ret.push(Darling::new(i, j));
    }
}

fn ferm2_test(
    freq: &Dvec,
    i: usize,
    j: usize,
    k: usize,
    f3qcm: &F3qcm,
    ret: &mut Vec<Fermi2>,
) -> bool {
    let diff1 = freq[i] - freq[j] - freq[k];
    let diff2 = -freq[i] + freq[j] - freq[k];
    let diff3 = -freq[i] - freq[j] + freq[k];
    let diff4 = freq[i] + freq[j] + freq[k];
    let delta = diff1 * diff2 * diff3 * diff4;
    let dalet = aminjm(diff1, diff2, diff3);
    if delta.abs() <= DLTOL {
        if f3qcm[(i, j, k)].abs() >= F3TOL {
            ret.push(Fermi2::new(i, j, k));
            return true;
        }
    } else if dalet.abs() <= DFTOL && f3qcm[(i, j, k)].abs() >= F3TOL {
        ret.push(Fermi2::new(i, j, k));
        return true;
    }
    false
}

fn aminjm(diff1: f64, diff2: f64, diff3: f64) -> f64 {
    let mut dalet = [diff1, diff2, diff3];
    dalet.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dalet[2]
}

impl Display for Rotor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Rotor::Diatomic => "diatomic",
                Rotor::Linear => "linear",
                Rotor::SphericalTop => "a spherical top",
                Rotor::OblateSymmTop => "an oblate symmetric top",
                Rotor::ProlateSymmTop => "a prolate symmetric top",
                Rotor::AsymmTop => "an asymmetric top",
                Rotor::None => unset_rotor!(),
            }
        )
    }
}