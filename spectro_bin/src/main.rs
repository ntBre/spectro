use std::path::Path;

use clap::Parser;
use spectro::Spectro;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// Optional name to operate on
    #[arg(value_parser)]
    infile: String,

    /// Writes the output in JSON format for use by other programs
    #[arg(short, long, value_parser, default_value_t = false)]
    json: bool,
}

fn main() -> Result<(), std::io::Error> {
    let cfg = Config::parse();
    let spectro = Spectro::load(&cfg.infile);
    let infile = Path::new(&cfg.infile);
    let dir = infile.parent().unwrap_or_else(|| Path::new("."));
    let (got, _) = spectro.run_files(
        dir.join("fort.15"),
        dir.join("fort.30"),
        dir.join("fort.40"),
    );
    if cfg.json {
        let data = serde_json::to_string_pretty(&got)?;
        println!("{data}");
    } else {
        spectro.write_output(&mut std::io::stdout(), &got)?;
    }
    Ok(())
}
