use spectro::Spectro;

fn main() -> Result<(), std::io::Error> {
    let spectro = Spectro::load("testfiles/c3h2/spectro.in");
    let got = spectro.run(
        "testfiles/c3h2/fort.15",
        "testfiles/c3h2/fort.30",
        "testfiles/c3h2/fort.40",
    );
    spectro.write_output(&mut std::io::stdout(), got)
}
