#[macro_use]
extern crate log;
use env_logger::{Builder, Env};

use c2c7demo::{draw, PointData};

use std::fs::File;
use std::io::BufReader;

use clap::Parser;
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input json filename
    input: String,
    /// Output image filename. PNG or SVG.
    output: String,

    /// Background image
    #[clap(short, long)]
    background: Option<String>,

    /// Specify once or twice to set log level info or debug repsectively.
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let log_level = if args.verbose == 0 {
        "error"
    } else if args.verbose == 1 {
        "info"
    } else {
        "debug"
    };
    let env = Env::default().filter_or("LOG_LEVEL", log_level);
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Seconds))
        .init();

    let file = File::open(args.input)?;
    let reader = BufReader::new(file);
    let json_data: PointData = serde_json::from_reader(reader)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    let background = args.background.map(|name| image::open(name).unwrap());
    let svg_output = args.output.ends_with(".svg");
    let document = draw(json_data, background, !svg_output);
    if svg_output {
        debug!("Save SVG");
        svg::save(args.output, &document).unwrap();
    } else {
        let mut opt = usvg::Options::default();
        opt.fontdb.load_system_fonts();
        let rtree = usvg::Tree::from_str(&document.to_string(), &opt.to_ref()).unwrap();
        let pixmap_size = rtree.svg_node().size.to_screen_size();
        let mut pixmap = tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height()).unwrap();
        debug!("Render");
        resvg::render(
            &rtree,
            usvg::FitTo::Original,
            tiny_skia::Transform::default(),
            pixmap.as_mut(),
        )
        .unwrap();
        image::RgbaImage::from_raw(pixmap.width(), pixmap.height(), pixmap.data().into())
            .unwrap()
            .save(args.output)
            .unwrap();
    }
    Ok(())
}
