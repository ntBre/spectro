#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Axis {
    #[default]
    A = 0,
    B = 1,
    C = 2,
}

/// coriolis resonance wᵢ = wⱼ
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coriolis {
    pub i: usize,
    pub j: usize,
    pub axis: Axis,
}

impl Coriolis {
    pub fn new(i: usize, j: usize, axis: usize) -> Self {
        Self {
            i,
            j,
            axis: match axis {
                0 => Axis::A,
                1 => Axis::B,
                2 => Axis::C,
                _ => panic!(),
            },
        }
    }
}

/// type 1 Fermi resonance 2wᵢ = wⱼ
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fermi1 {
    pub i: usize,
    pub j: usize,
}

impl Fermi1 {
    pub fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}

/// type 2 Fermi resonance wₖ = wⱼ + wᵢ
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fermi2 {
    pub i: usize,
    pub j: usize,
    pub k: usize,
}

impl Fermi2 {
    pub fn new(i: usize, j: usize, k: usize) -> Self {
        Self { i, j, k }
    }
}

/// Darling-Dennison resonance 2wᵢ = 2wⱼ
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Darling {
    pub i: usize,
    pub j: usize,
}

impl Darling {
    pub fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}
