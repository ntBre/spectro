use spectro::Spectro;

fn main() {
    let spectro = Spectro::load("testfiles/h2o/spectro.in");
    let got = spectro.run(
        "testfiles/h2o/fort.15",
        "testfiles/h2o/fort.30",
        "testfiles/h2o/fort.40",
    );
    dbg!(got);
}
