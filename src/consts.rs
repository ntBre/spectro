use std::f64::consts::PI;

pub(crate) const _HE: f64 = 4.359813653;

pub(crate) const _A0: f64 = 0.52917706;

/// HE / (AO * AO) from fortran. something about hartrees and AO is bohr radius
pub(crate) const FACT2: f64 = 4.359813653 / (0.52917706 * 0.52917706);

pub(crate) const FUNIT3: f64 =
    4.359813653 / (0.52917706 * 0.52917706 * 0.52917706);

pub(crate) const FUNIT4: f64 =
    4.359813653 / (0.52917706 * 0.52917706 * 0.52917706 * 0.52917706);

pub(crate) const _AMU: f64 = 1.66056559e-27;

pub(crate) const _ELMASS: f64 = 0.91095344e-30;

/// looks like cm-1 to mhz factor
pub(crate) const CL: f64 = 2.99792458;

pub(crate) const ALAM: f64 = 4.0e-2 * (PI * PI * CL) / (PH * AVN);

/// constant for converting moments of inertia to rotational constants
pub(crate) const CONST: f64 = 1.0e+02 * (PH * AVN) / (8.0e+00 * PI * PI * CL);

/// planck's constant in atomic units?

pub(crate) const PH: f64 = 6.626176;

/// pre-computed sqrt of ALAM
pub(crate) const SQLAM: f64 = 0.172_221_250_379_107_6;

pub(crate) const FACT3: f64 = 1.0e6 / (SQLAM * SQLAM * SQLAM * PH * CL);

pub(crate) const FACT4: f64 = 1.0e6 / (ALAM * ALAM * PH * CL);

/// ALPHA_CONST IS THE PI*SQRT(C/H) FACTOR
pub(crate) const ALPHA_CONST: f64 = 0.086112;

/// PRINCIPAL ---> CARTESIAN
pub(crate) static IPTOC: nalgebra::Matrix3x6<usize> = nalgebra::matrix![
    2,    1,    0,    2,    0,    1;
    0,    2,    1,    1,    2,    0;
    1,    0,    2,    0,    1,    2;
];

/// CARTESIAN---> PRINCIPAL
pub(crate) static ICTOP: nalgebra::Matrix3x6<usize> = nalgebra::matrix![
    1,    2,    0,    2,    0,    1;
    2,    0,    1,    1,    2,    0;
    0,    1,    2,    0,    1,    2;
];

/// avogadro's number
pub(crate) const AVN: f64 = 6.022045;

pub(crate) const _PARA: f64 = 1.0 / AVN;

// pre-compute the sqrt and make const
pub(crate) const SQRT_AVN: f64 = 2.453_985_533_779_692;

// conversion to cm-1
pub(crate) const WAVE: f64 = 1e4 * SQRT_AVN / (2.0 * PI * CL);
