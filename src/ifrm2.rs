use std::collections::HashMap;

pub(crate) struct Ifrm2(HashMap<(usize, usize), usize>);

impl Ifrm2 {
    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }

    pub(crate) fn insert(
        &mut self,
        k: (usize, usize),
        v: usize,
    ) -> Option<usize> {
        self.0.insert(k, v)
    }

    pub(crate) fn get(&self, k: &(usize, usize)) -> Option<&usize> {
        self.0.get(k)
    }

    /// check if `k` is contained in `self` and its value is `v`
    pub(crate) fn check(&self, k: (usize, usize), v: usize) -> bool {
        let tmp = self.get(&k);
        tmp.is_some() && *tmp.unwrap() == v
    }
}

impl std::fmt::Debug for Ifrm2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for ((a, b), v) in &self.0 {
            writeln!(f, "({:5}, {:5}) => {:5}", a, b, v)?;
        }
        Ok(())
    }
}
