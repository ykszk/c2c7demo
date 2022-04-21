use std::fs::{self, File};
use std::io::Read;
#[macro_use]
extern crate log;
use env_logger::Env;
use tract_onnx::prelude::tract_ndarray as ndarray;

use c2c7demo::{extract_points, PointData, LABELS};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Args)]
struct DefaultArgs {
    /// Input filename
    input: String,

    /// Input shape. e.g. "6,768,768"
    shape: String,

    /// Output filename
    output: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract peak points from heatmaps
    Peak(DefaultArgs),
    /// Convert raw heatmap to RGB image
    Heatmap(DefaultArgs),
    /// Extract affinity map
    Affinity(DefaultArgs),
}

type Shape3D = (usize, usize, usize);

fn parse_shape(shape: &str) -> Shape3D {
    let shape: Vec<usize> = shape
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    assert_eq!(shape.len(), 3);
    (shape[0], shape[1], shape[2])
}

fn load_raw(filename: &str, shape: &Shape3D) -> ndarray::Array3<u8> {
    let mut f = File::open(&filename).expect("no file found");
    let metadata = fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read_exact(&mut buffer).expect("buffer overflow");
    let (chs, height, width) = shape;
    if buffer.len() != chs * height * width {
        panic!("Invalid input size")
    } else {
        debug!("{} loaded.", filename);
    }
    ndarray::Array3::from_shape_vec(*shape, buffer).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = Env::default().filter_or("LOG_LEVEL", "debug");
    env_logger::init_from_env(env);
    let cli = Cli::parse();
    match &cli.command {
        Commands::Peak(DefaultArgs {
            input,
            shape,
            output,
        }) => {
            let filename = input;
            let shape = parse_shape(shape);
            let (_chs, height, width) = shape;
            let arr = load_raw(filename, &shape);
            debug!("array shape: {:?}", arr.shape());
            let optimal_points = extract_points(&arr);
            debug!("Optimal points {:?}", optimal_points);
            let labels: Vec<String> = LABELS.iter().map(|s| String::from(*s)).collect();
            let data = PointData::new(&optimal_points, &labels, width, height, "");
            data.save(output)?;
        }
        Commands::Heatmap(DefaultArgs {
            input,
            shape,
            output,
        }) => {
            let filename = input;
            let shape = parse_shape(shape);
            let arr = load_raw(filename, &shape);
            let img = c2c7demo::extract_heatmap(&arr)?;
            img.save(output).unwrap();
        }
        Commands::Affinity(DefaultArgs {
            input,
            shape,
            output,
        }) => {
            let filename = input;
            let shape = parse_shape(shape);
            let arr = load_raw(filename, &shape);
            let img = c2c7demo::extract_affinity(&arr)?;
            img.save(output).unwrap();
        }
    }
    Ok(())
}
