use std::{error::Error, path::Path};

use clap::Parser;
use spectro::{Spectro, SpectroFinish};

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Optional name to operate on
    #[arg(value_parser)]
    infile: String,

    /// Writes the output in JSON format for use by other programs
    #[arg(short, long, value_parser, default_value_t = false)]
    json: bool,

    /// finish a run by loading a [SpectroFinish] from a JSON file
    #[arg(short, long, value_parser, default_value_t = false)]
    finish: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = Args::parse();
    let (got, spectro) = if cfg.finish {
        let SpectroFinish {
            spectro,
            freq,
            f3qcm,
            f4qcm,
            irreps,
            lxm,
            lx,
        } = SpectroFinish::load(&cfg.infile)?;
        let (g, _) = spectro.finish(freq, f3qcm, f4qcm, irreps, lxm, lx);
        (g, spectro)
    } else {
        let spectro = Spectro::load(&cfg.infile);
        let infile = Path::new(&cfg.infile);
        let dir = infile.parent().unwrap_or_else(|| Path::new("."));
        let (g, _) = spectro.run_files(
            dir.join("fort.15"),
            dir.join("fort.30"),
            dir.join("fort.40"),
        );
        (g, spectro)
    };
    if cfg.json {
        let data = serde_json::to_string_pretty(&got)?;
        println!("{data}");
    } else {
        spectro.write_output(&mut std::io::stdout(), &got)?;
    }
    Ok(())
}
