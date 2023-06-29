use crate::Spectro;

impl Default for Spectro {
    fn default() -> Self {
        Self {
            header: Default::default(),
            geom: Default::default(),
            weights: Default::default(),
            curvils: Default::default(),
            degmodes: Default::default(),
            dummies: Default::default(),
            rotor: Default::default(),
            n3n: Default::default(),
            i3n3n: Default::default(),
            i4n3n: Default::default(),
            nvib: Default::default(),
            i2vib: Default::default(),
            i3vib: Default::default(),
            i4vib: Default::default(),
            natom: Default::default(),
            axes: Default::default(),
            primat: Default::default(),
            rotcon: Default::default(),
            iatom: Default::default(),
            axis_order: Default::default(),
            axis: Default::default(),
            verbose: Default::default(),
            symm_tol: 1e-6,
        }
    }
}
