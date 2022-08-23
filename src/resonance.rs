/// coriolis resonance wᵢ = wⱼ
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coriolis {
    pub i: usize,
    pub j: usize,
}

impl Coriolis {
    pub fn new(i: usize, j: usize) -> Self {
        Self { i, j }
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
