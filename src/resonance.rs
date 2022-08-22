/// coriolis resonance wᵢ = wⱼ
pub struct Coriolis {
    pub i: usize,
    pub j: usize,
}

impl Coriolis {
    pub fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}
