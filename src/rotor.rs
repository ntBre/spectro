use std::fmt::Display;

use crate::resonance::{Coriolis, Darling, Fermi1, Fermi2};
use crate::utils::find3r;
use crate::Dvec;
type Tensor3 = tensor::tensor3::Tensor3<f64>;

/// cutoff for determining if moments of inertia are equal for rotor
/// classification
pub(crate) const ROTOR_EPS: f64 = 1.0e-4;

// TODO probably need to get rid of default here. it will complicate the spectro
// usage a little bit but then I don't have to have the None variant and keep
// checking for it
#[derive(Clone, Debug, Default, PartialEq)]
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

impl Rotor {
    /// panics if `self` is not set
    pub fn is_sym_top(&self) -> bool {
        use Rotor::*;
        match &self {
            Diatomic => false,
            Linear => false,
            SphericalTop => false,
            OblateSymmTop => true,
            ProlateSymmTop => true,
            AsymmTop => false,
            None => panic!("rotor type not set"),
        }
    }

    /// panics if `self` is not set
    pub fn is_prolate(&self) -> bool {
        assert!(*self != Rotor::None);
        *self == Rotor::ProlateSymmTop
    }

    pub fn coriolis(
        &self,
        n1dm: usize,
        i1mode: &[usize],
        freq: &Dvec,
        zmat: &Tensor3,
    ) -> Vec<Coriolis> {
        // tolerances for resonance checking
        const CTOL: f64 = 200.0;
        const ZTOL: f64 = 0.25;

        let mut ret = Vec::new();
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::Linear => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop => todo!(),
            Rotor::ProlateSymmTop => todo!(),
            Rotor::AsymmTop => {
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
            Rotor::None => todo!(),
        }
        ret
    }

    pub fn fermi1(
        &self,
        n1dm: usize,
        i1mode: &[usize],
        freq: &Dvec,
        f3qcm: &[f64],
    ) -> Vec<Fermi1> {
        // tolerances for resonance checking
        const FTOL1: f64 = 200.0;
        const F3TOL: f64 = 10.0;

        let mut ret = Vec::new();
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::Linear => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop => todo!(),
            Rotor::ProlateSymmTop => todo!(),
            Rotor::AsymmTop => {
                for ii in 0..n1dm {
                    // loop over i1mode?
                    let i = i1mode[ii];
                    for jj in 0..n1dm {
                        let j = i1mode[jj];
                        let diff = 2.0 * freq[i] - freq[j];
                        if diff.abs() <= FTOL1 {
                            let iij = find3r(i, i, j);
                            if f3qcm[iij].abs() >= F3TOL {
                                ret.push(Fermi1::new(i, j));
                            }
                        }
                    }
                }
            }
            Rotor::None => todo!(),
        }
        ret
    }

    pub fn fermi2(
        &self,
        n1dm: usize,
        i1mode: &[usize],
        freq: &Dvec,
        f3qcm: &[f64],
    ) -> Vec<Fermi2> {
        // tolerances for resonance checking
        const F3TOL: f64 = 10.0;
        const DLTOL: f64 = 1000.0;
        const DFTOL: f64 = 200.0;

        let mut ret = Vec::new();
        // I wonder if this match is even necessary without computing the
        // perturbation. I think that might only be since the perturbation uses
        // rotcon which does vary in shape between rotor types
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::Linear => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop => todo!(),
            Rotor::ProlateSymmTop => todo!(),
            Rotor::AsymmTop => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        for kk in 0..jj {
                            let k = i1mode[kk];
                            let diff1 = freq[i] - freq[j] - freq[k];
                            let diff2 = -freq[i] + freq[j] - freq[k];
                            let diff3 = -freq[i] - freq[j] + freq[k];
                            let diff4 = freq[i] + freq[j] + freq[k];
                            let delta = diff1 * diff2 * diff3 * diff4;
                            let ijk = find3r(i, j, k);
                            if i != j && j != k && i != k {
                                // min of the 3 diffs, but they take the abs in
                                // the sort, so take the last element instead of
                                // first and use abs in the tolerance later
                                let mut dalet = [diff1, diff2, diff3];
                                dalet
                                    .sort_by(|a, b| a.partial_cmp(&b).unwrap());
                                let dalet = dalet[2];
                                if delta.abs() <= DLTOL {
                                    if f3qcm[ijk].abs() >= F3TOL {
                                        ret.push(Fermi2::new(i, j, k));
                                        continue;
                                    }
                                } else {
                                    if dalet.abs() <= DFTOL {
                                        if f3qcm[ijk].abs() >= F3TOL {
                                            ret.push(Fermi2::new(i, j, k));
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Rotor::None => todo!(),
        }
        ret
    }

    pub fn darling(
        &self,
        n1dm: usize,
        i1mode: &[usize],
        freq: &Dvec,
    ) -> Vec<Darling> {
        // tolerances for resonance checking
        const DDTOL: f64 = 300.0;

        let mut ret = Vec::new();
        match self {
            Rotor::Diatomic => todo!(),
            Rotor::Linear => todo!(),
            Rotor::SphericalTop => todo!(),
            Rotor::OblateSymmTop => todo!(),
            Rotor::ProlateSymmTop => todo!(),
            Rotor::AsymmTop => {
                for ii in 1..n1dm {
                    let i = i1mode[ii];
                    for jj in 0..ii {
                        let j = i1mode[jj];
                        let diff = 2.0 * freq[i] - 2.0 * freq[j];
                        if diff.abs() <= DDTOL {
                            ret.push(Darling::new(i, j));
                        }
                    }
                }
            }
            Rotor::None => todo!(),
        }
        ret
    }
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
                Rotor::None => panic!("rotor type not set"),
            }
        )
    }
}
