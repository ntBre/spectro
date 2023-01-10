use std::fmt::Display;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum State {
    /// singly-degenerate mode state
    I1st(Vec<usize>),

    /// doubly-degenerate mode state. the first element of each tuple
    /// corresponds to the entry in `i2sts` and the second to `ilsts` in the
    /// fortran code
    I2st(Vec<(usize, usize)>),

    /// triply-degenerate mode state
    I3st(Vec<usize>),

    /// combination band of a singly-degenerate mode and a doubly-degenerate
    /// mode
    I12st {
        i1st: Vec<usize>,
        i2st: Vec<(usize, usize)>,
    },
}

impl State {
    pub fn len(&self) -> usize {
        match self {
            State::I1st(v) => v.len(),
            State::I2st(v) => v.len(),
            State::I3st(v) => v.len(),
            State::I12st { i1st, i2st } => {
                assert_eq!(i1st.len(), i2st.len());
                i1st.len()
            }
        }
    }

    /// return vectors of the separated singly-degenerate, doubly-degenerate,
    /// and triply-degenerate states. these are referrred to in the Fortran code
    /// as `i1sts`, `i2sts`, and `i3sts`.
    #[allow(clippy::type_complexity)]
    pub fn partition(
        states: &[Self],
    ) -> (Vec<Vec<usize>>, Vec<Vec<(usize, usize)>>, Vec<Vec<usize>>) {
        let mut ret = (vec![], vec![], vec![]);
        for s in states {
            match &s {
                State::I1st(v) => {
                    ret.0.push(v.clone());

                    ret.1.push(vec![(0, 0); v.len()]);
                    ret.2.push(vec![0; v.len()]);
                }
                State::I2st(v) => {
                    ret.1.push(v.clone());

                    ret.0.push(vec![0; v.len()]);
                    ret.2.push(vec![0; v.len()]);
                }
                State::I3st(v) => {
                    ret.2.push(v.clone());

                    ret.0.push(vec![0; v.len()]);
                    ret.1.push(vec![(0, 0); v.len()]);
                }
                State::I12st { i1st, i2st } => {
                    ret.0.push(i1st.clone());
                    ret.1.push(i2st.clone());
                    ret.2.push(vec![0; i1st.len()]);
                }
            }
        }
        ret
    }
}

// this is actually awful but it works
pub struct I1states(pub Vec<Vec<usize>>);

impl Display for I1states {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for state in &self.0 {
            write!(f, "{}", State::I1st(state.clone()))?;
        }
        Ok(())
    }
}

// this is actually awful but it works
pub struct I2states(pub Vec<Vec<(usize, usize)>>);

impl Display for I2states {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, state) in self.0.iter().enumerate() {
            writeln!(f, "{i}:{}", State::I2st(state.clone()))?;
        }
        Ok(())
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            State::I1st(v) => {
                for s in v {
                    write!(f, "{s:5}")?
                }
            }
            State::I2st(v) => {
                for (a, b) in v {
                    write!(f, "({a},{b})")?
                }
            }
            State::I3st(_) => todo!(),
            State::I12st { i1st: _, i2st: _ } => todo!(),
        }
        Ok(())
    }
}
