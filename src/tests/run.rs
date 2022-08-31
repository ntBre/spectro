use std::{fs::read_to_string, path::PathBuf};

use approx::{abs_diff_eq, assert_abs_diff_eq};
use serde::Deserialize;

use crate::*;

use na::DVector;
use nalgebra as na;

fn load_want(filename: PathBuf) -> Output {
    #[derive(Clone, Deserialize, Debug)]
    struct Want {
        harms: Vec<f64>,
        funds: Vec<f64>,
        corrs: Vec<f64>,
        rots: Vec<Vec<f64>>,
    }
    let data = read_to_string(filename).unwrap();
    let want: Want = serde_json::from_str(&data).unwrap();
    let mut rots = vec![Rot::new(
        vec![0; want.harms.len()],
        want.rots[0][2],
        want.rots[0][0],
        want.rots[0][1],
    )];
    for i in 0..want.harms.len() {
        let mut tmp = vec![0; want.harms.len()];
        tmp[i] = 1;
        rots.push(Rot::new(
            tmp,
            want.rots[i + 1][2],
            want.rots[i + 1][0],
            want.rots[i + 1][1],
        ));
    }
    Output {
        harms: Dvec::from(want.harms),
        funds: want.funds,
        corrs: want.corrs,
        rots,
    }
}

#[derive(Clone)]
struct Test {
    infile: PathBuf,
    fort15: PathBuf,
    fort30: PathBuf,
    fort40: PathBuf,
    want: Output,
}

impl Test {
    fn new(dir: &'static str) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: start.join(dir).join("spectro.in"),
            fort15: start.join(dir).join("fort.15"),
            fort30: start.join(dir).join("fort.30"),
            fort40: start.join(dir).join("fort.40"),
            want: load_want(start.join(dir).join("summary.json")),
        }
    }
}

fn check(got: Vec<f64>, want: Vec<f64>, msg: &str, label: &str) {
    assert!(got.len() == want.len());
    if !abs_diff_eq!(
        Dvec::from(got.clone()),
        Dvec::from(want.clone()),
        epsilon = 0.1
    ) {
        println!("\n{:>5}{:>8}{:>8}{:>8}", "Mode", "Got", "Want", "Diff");
        for i in 0..got.len() {
            println!(
                "{:5}{:8.1}{:8.1}{:8.1}",
                i + 1,
                got[i],
                want[i],
                got[i] - want[i],
            );
        }
        assert!(false, "{} differ at {}", msg, label);
    }
}

fn check_rots(got: Vec<Rot>, want: Vec<Rot>, infile: &str) {
    assert_eq!(got.len(), want.len());
    if !abs_diff_eq!(
        DVector::from(got.clone()),
        DVector::from(want.clone()),
        epsilon = 3e-5
    ) {
        println!("{}", "got");
        for g in &got {
            println!("{}", g);
        }
        println!("{}", "want");
        for g in &want {
            println!("{}", g);
        }
        println!("{}", "diff");
        for g in 0..got.len() {
            println!(
                "{}",
                Rot::new(
                    got[g].state.clone(),
                    got[g].a - want[g].a,
                    got[g].b - want[g].b,
                    got[g].c - want[g].c,
                )
            );
        }
        assert!(false, "rots differ at {}", infile);
    }
}

#[test]
fn test_run() {
    let tests = [
        // asymmetric tops
        Test::new("h2o"),
        Test::new("h2co"),
        Test::new("c3h2"),
        Test::new("c2h4"),
        Test::new("c4h3+"),
        Test::new("allyl"),
        Test::new("h2o_sic"),
        Test::new("c3hf"),
        Test::new("c3hcl"),
        Test::new("c3hcn"),
        Test::new("c3hcn010"),
        Test::new("c-hoco"),
        Test::new("hno"),
        Test::new("hocn"),
        Test::new("hoco+"),
        Test::new("hpsi"),
        Test::new("hso"),
        Test::new("hss"),
        Test::new("nh2-"),
        Test::new("nnoh+"),
        Test::new("sic2"),
        Test::new("t-hoco"),
        // symmetric tops
        // Test::new("nh3"),

        // linear
        // Test::new("c2h-"),
        // Test::new("hcn"),
        // Test::new("hco+"),
        // Test::new("hmgnc"),
        // Test::new("hnc"),
    ];
    for test in Vec::from(&tests[..]) {
        let infile = test.infile.to_str().unwrap();

        let spectro = Spectro::load(infile);
        let got = spectro.run(test.fort15, test.fort30, test.fort40);

        assert_abs_diff_eq!(got.harms, test.want.harms, epsilon = 0.1);
        check(got.funds, test.want.funds, "funds", infile);
        check(got.corrs, test.want.corrs, "corrs", infile);
        check_rots(got.rots, test.want.rots, infile);
    }
}
