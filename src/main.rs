use std::path::Path;

use spectro::Spectro;

fn main() -> Result<(), std::io::Error> {
    let args: Vec<_> = std::env::args().collect();
    let infile = match args.get(1) {
        Some(f) => f,
        // in this case, dir also just becomes .
        None => todo!("need to read from stdin"),
    };
    let spectro = Spectro::load(infile);
    let infile = Path::new(infile);
    let dir = infile.parent().unwrap_or_else(|| Path::new("."));
    let got = spectro.run(
        dir.join("fort.15"),
        dir.join("fort.30"),
        dir.join("fort.40"),
    );
    spectro.write_output(&mut std::io::stdout(), got)
}
