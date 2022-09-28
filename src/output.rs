use std::fmt::Display;

use crate::{quartic::Quartic, rot::Rot, sextic::Sextic};

/// contains all of the output data from running Spectro
#[derive(Clone, Debug)]
pub struct Output {
    /// harmonic frequencies
    pub harms: Vec<f64>,

    /// partially resonance-corrected anharmonic frequencies
    pub funds: Vec<f64>,

    /// fully resonance-corrected anharmonic frequencies
    pub corrs: Vec<f64>,

    /// vibrationally averaged rotational constants
    pub rots: Vec<Rot>,

    /// quartic distortion coefficients
    pub quartic: Quartic,

    /// sextic distortion coefficients
    pub sextic: Sextic,
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Vibrational Frequencies (cm-1):\n{:>5}{:>8}{:>8}{:>8}",
            "Mode", "Harm", "Fund", "Corr"
        )?;
        for i in 0..self.harms.len() {
            writeln!(
                f,
                "{:5}{:8.1}{:8.1}{:8.1}",
                i + 1,
                self.harms[i],
                self.funds[i],
                self.corrs[i]
            )?;
        }

        writeln!(f, "\nRotational Constants (cm-1):")?;
        let width = 5 * self.rots[0].state.len();
        writeln!(f, "{:^width$}{:^12}{:^12}{:^12}", "State", "A", "B", "C")?;
        for rot in &self.rots {
            writeln!(f, "{}", rot)?;
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
