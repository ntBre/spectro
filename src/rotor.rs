use std::fmt::Display;

use tensor::Tensor3;

use crate::resonance::Coriolis;
use crate::Dvec;

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

    pub fn coriolis_resonances(
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
                                if zmat[(i, j, z)] >= ZTOL {
                                    ret.push(Coriolis::new(i, j));
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
