use std::fmt::Display;

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// singly-degenerate mode state
    I1st(Vec<usize>),

    /// doubly-degenerate mode state
    I2st(Vec<usize>),

    /// triply-degenerate mode state
    I3st(Vec<usize>),

    /// combination band of a singly-degenerate mode and a doubly-degenerate
    /// mode
    I12st { i1st: Vec<usize>, i2st: Vec<usize> },
}

impl State {
    /// return vectors of the separated singly-degenerate, doubly-degenerate,
    /// and triply-degenerate states. these are referrred to in the Fortran code
    /// as `i1sts`, `i2sts`, and `i3sts`.
    pub fn partition(
        states: &[Self],
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut ret = (vec![], vec![], vec![]);
        for s in states {
            match &s {
                State::I1st(v) => {
                    ret.0.push(v.clone());

                    ret.1.push(vec![0; v.len()]);
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
                    ret.1.push(vec![0; v.len()]);
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
pub struct States(pub Vec<Vec<usize>>);

impl Display for States {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for state in &self.0 {
            write!(f, "{}", State::I1st(state.clone()))?;
        }
        Ok(())
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        match self {
            State::I1st(v) | State::I2st(v) => {
                for i in v {
                    write!(f, "{:5}", i)?;
                }
            }
            State::I3st(_) => todo!(),
            State::I12st { i1st: _, i2st: _ } => todo!(),
        }
        Ok(())
    }
}
