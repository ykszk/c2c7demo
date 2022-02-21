use std::error::Error;
use std::path::PathBuf;

#[macro_use]
extern crate log;
use c2c7demo::{draw, extract_points, PointData, LABELS};
use clahe::clahe;
use clap::Parser;
use env_logger::{Builder, Env};
use image::error::{ImageFormatHint, UnsupportedError};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use serde::Serialize;
use std::fs;
use tract_onnx::prelude::tract_ndarray as ndarray;
use tract_onnx::prelude::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input_dir: String,
    /// Output image filename
    output_dir: Option<PathBuf>,

    /// Model path
    #[clap(short, long, default_value = c2c7demo::DEFAULT_MODEL)]
    model: String,
    /// Output CSV filename
    #[clap(short, long, default_value = "c2c7.csv")]
    csv: String,
    /// Specify once or twice to set log level info or debug repsectively.
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct TableRow {
    input: String,
    output: String,
    C2A_x: f32,
    C2A_y: f32,
    C2P_x: f32,
    C2P_y: f32,
    C7A_x: f32,
    C7A_y: f32,
    C7P_x: f32,
    C7P_y: f32,
    cobb_angle: f32,
}

impl TableRow {
    pub fn new(input: String, output: String, points: &[c2c7demo::Point]) -> Self {
        let cobb_angle = c2c7demo::calc_angle(points);
        TableRow {
            input,
            output,
            C2A_x: points[0].0,
            C2A_y: points[0].1,
            C2P_x: points[1].0,
            C2P_y: points[1].1,
            C7A_x: points[2].0,
            C7A_y: points[2].1,
            C7P_x: points[3].0,
            C7P_y: points[3].1,
            cobb_angle,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
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

    if !fs::metadata(&args.input_dir).unwrap().is_dir() {
        panic!("Input is not a directory.");
    }
    let output_dir = match args.output_dir {
        Some(pbuf) => pbuf,
        None => {
            info!("Use input directory as the output directory.");
            PathBuf::from(&args.input_dir)
        }
    };
    if !fs::metadata(&output_dir).unwrap().is_dir() {
        panic!("Output is not a directory.")
    }

    let model_path = c2c7demo::resolve_model_path(&args.model)?;

    let mut table = Vec::new();
    for entry in fs::read_dir(args.input_dir)? {
        let entry = entry?;
        let input = entry.path();
        if input.is_dir() {
            info!("Skip directory {:?}", input);
            continue
        }
        let input_str = input.to_str().unwrap();
        let try_dicom = if let Some(ext) = input.extension() {
            info!("Input image extension {:?}", ext);
            ext == "dcm" || ext == "dicom"
        } else {
            false
        };

        info!("Open {:?}", input);
        let original_img = if try_dicom {
            info!("Open as dicom");
            let dicom_img = c2c7demo::load_dicom(input_str);
            match dicom_img {
                Ok(img) => img,
                Err(e) => {
                    debug!("{:?}", e.to_string());
                    image::open(&input).unwrap()
                }
            }
        } else {
            image::open(&input).unwrap()
        };
        let target_height = c2c7demo::TARGET_HEIGHT as u32;
        let (width, height) = original_img.dimensions();
        let gray_img = original_img.grayscale();
        let new_width: u32 =
            ((width as f64) * (target_height as f64) / (height as f64)).round() as u32;
        debug!("Original image size: {} {}", width, height);
        info!("Resize to w{} h{}", new_width, target_height);
        let resized_img = gray_img.resize_exact(new_width, target_height, FilterType::Triangle);
        let img = match &resized_img {
            DynamicImage::ImageLuma8(img) => {
                debug!("input u8 to clahe");
                clahe(img, 32, 32, 10)?
            }
            DynamicImage::ImageLuma16(img) => {
                debug!("input u16 to clahe");
                clahe(img, 32, 32, 10000)?
            }
            _ => {
                let hint = ImageFormatHint::Name("u8 or u16 image is expected".to_string());
                return Err(UnsupportedError::from(hint).into());
            }
        };

        let pad_width = ((new_width as f64 / 256.0).ceil() * 256.0) as u32;
        let (input_width, input_height) = (pad_width, target_height);
        let batch_size = 2;

        info!("Load model");
        let model = tract_onnx::onnx()
            .model_for_path(&model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(batch_size, 1, input_height as usize, input_width as usize),
                ),
            )?
            .into_optimized()?
            .into_runnable()?;

        let input_tensor: Tensor = c2c7demo::create_input_batch(&img, input_width, input_height).into();

        debug!("Input shape {:?}", input_tensor.shape());
        info!("Run model");
        // run the model on the input
        let result = model.run(tvec!(input_tensor))?;
        debug!("Output shape {:?}", result[0].shape());
        let (img_width, img_height) = img.dimensions();
        let arr4: ndarray::ArrayView4<f32> = result[0]
            .to_array_view::<f32>()?
            .into_dimensionality()
            .unwrap();
        let best_batch: usize = c2c7demo::choose_best_batch(&arr4) as _;
        let flip_needed = best_batch != 0;

        let arr3 = arr4
            .slice(ndarray::s![
                best_batch,
                ..,
                0..img_height as usize,
                0..img_width as usize
            ])
            .map(|v| (v * 255.0) as u8);

        let optimal_points = extract_points(&arr3);
        let optimal_points = if flip_needed {
            optimal_points
                .into_iter()
                .map(|(x, y)| (img_width as f32 - x - 1.0, y))
                .collect()
        } else {
            optimal_points
        };

        debug!("Optimal points {:?}", optimal_points);
        let labels: Vec<String> = LABELS.iter().map(|s| String::from(*s)).collect();
        let data = PointData::new(
            &optimal_points,
            &labels,
            img_width as _,
            img_height as _,
            "",
        );

        let background = match gray_img {
            DynamicImage::ImageLuma8(img) => Some(DynamicImage::ImageLuma8(img)),
            DynamicImage::ImageLuma16(img) => Some(c2c7demo::luma16toluma8(&img)),
            _ => None,
        };
        let document = draw(data, background, false);
        let mut output = output_dir.clone();
        output.push(input.clone().with_extension("svg").file_name().unwrap());
        info!("Save {:?}", output);
        svg::save(&output, &document).unwrap();

        // scale coordinates
        let scale = original_img.dimensions().1 as f32 / c2c7demo::TARGET_HEIGHT as f32;
        let scaled_points: Vec<c2c7demo::Point> = optimal_points
            .into_iter()
            .map(|(x, y)| (scale * x, scale * y))
            .collect();
        let row = TableRow::new(
            input.to_str().unwrap().into(),
            fs::canonicalize(output)?.to_str().unwrap().into(),
            &scaled_points,
        );
        table.push(row);
    }
    let mut csv_filename = output_dir;
    csv_filename.push(args.csv);
    info!("Save csv {:?}", csv_filename);
    let mut wrt = csv::Writer::from_path(csv_filename)?;
    table
        .into_iter()
        .for_each(|row| wrt.serialize(row).unwrap());
    wrt.flush()?;

    Ok(())
}
