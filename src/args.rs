use std::path::{PathBuf};

use argh::FromArgs;

#[derive(FromArgs)]
/// Axum-T5 server written in Rust
pub struct Args {
    /// listen on 0.0.0.0
    #[argh(switch, short = 'l')]
    pub listen: bool,

    /// port number
    #[argh(option, short = 'p', default = "9999")]
    pub port: usize,

    /// number of workers, more workers means higher VRAM usage but higher throughput
    #[argh(option, short = 'w', default = "2")]
    pub no_workers: usize,

    /// path to model, has to be a directory
    #[argh(option, short = 'm')]
    pub model: PathBuf,
}