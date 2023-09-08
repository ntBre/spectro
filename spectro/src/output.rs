use std::fmt::Display;

use serde::{Deserialize, Serialize};
use symm::{Irrep, Molecule};

use crate::{quartic::Quartic, resonance::Restst, rot::Rot, sextic::Sextic};

/// contains all of the output data from running Spectro
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Output {
    /// harmonic frequencies
    pub harms: Vec<f64>,

    /// partially resonance-corrected anharmonic frequencies
    pub funds: Vec<f64>,

    /// fully resonance-corrected anharmonic frequencies
    pub corrs: Vec<f64>,

    /// vibrationally averaged rotational constants
    pub rots: Vec<Rot>,

    /// equilibrium rotational constants
    pub rot_equil: Vec<f64>,

    pub irreps: Vec<Irrep>,

    /// quartic distortion coefficients
    pub quartic: Quartic,

    /// sextic distortion coefficients
    pub sextic: Sextic,

    /// zero-point vibrational energy
    pub zpt: f64,

    pub geom: Molecule,

    pub lxm: Vec<Vec<f64>>,

    #[serde(default)]
    pub lx: Vec<Vec<f64>>,

    pub linear: bool,

    pub resonances: Restst,
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Geometry: {:.8}", self.geom)?;
        writeln!(
            f,
            "Vibrational Frequencies (cm-1):\n{:>5}{:>8}{:>8}{:>8}{:>8}",
            "Mode", "Symm", "Harm", "Fund", "Corr"
        )?;
        for i in 0..self.harms.len() {
            writeln!(
                f,
                "{:5}{:>8}{:8.1}{:8.1}{:8.1}",
                i + 1,
                self.irreps[i],
                self.harms[i],
                self.funds[i],
                self.corrs[i]
            )?;
        }

        writeln!(f, "\nRotational Constants (cm-1):")?;
        if !self.rots.is_empty() {
            let width = 5 * self.rots[0].state.len();
            if !self.linear {
                writeln!(
                    f,
                    "{:^width$}{:^12}{:^12}{:^12}",
                    "State", "A", "B", "C"
                )?;
                for rot in &self.rots {
                    writeln!(f, "{rot}")?;
                }
            } else {
                writeln!(f, "{:^width$}{:^12}", "State", "B")?;
                for rot in &self.rots {
                    writeln!(
                        f,
                        "{}{:12.7}",
                        rot.state,
                        rot.b + self.rot_equil[1]
                    )?;
                }
            }
        }

        writeln!(
            f,
            "\nQuartic Distortion Constants (cm-1):\n{}",
            self.quartic
        )?;

        writeln!(f, "Sextic Distortion Constants (cm-1):\n{}", self.sextic)?;

        Ok(())
    }
}
