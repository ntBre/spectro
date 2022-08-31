use super::*;

extern crate test;

#[bench]
fn bench_force4(b: &mut test::Bencher) {
    let s = Spectro::load("testfiles/h2o/spectro.in");
    let fc2 = load_fc2("testfiles/h2o/fort.15", s.n3n);
    let fc2 = s.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, &sqm);
    let (harms, lxm) = symm_eigen_decomp(fxm);
    let freq = to_wavenumbers(harms);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);
    let f4x = load_fc4("testfiles/h2o/fort.40", s.n3n);
    let mut f4x = s.rot4th(f4x, s.axes);
    b.iter(|| {
        force4(s.n3n, &mut f4x, &lx, s.nvib, &freq);
    });
}
