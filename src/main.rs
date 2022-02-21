use std::error::Error;
use std::path::Path;

#[macro_use]
extern crate log;
use c2c7demo::{draw, extract_points, PointData, LABELS};
use clahe::clahe;
use clap::Parser;
use env_logger::{Builder, Env};
use image::error::{ImageFormatHint, UnsupportedError};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use tract_onnx::prelude::tract_ndarray as ndarray;
use tract_onnx::prelude::*;

#[derive(clap::ArgEnum, Clone, Debug)]
enum FaceDirection {
    Left,
    Right,
    Auto,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input: String,
    /// Output image filename
    output: String,

    /// Model path
    #[clap(short, long, default_value = c2c7demo::DEFAULT_MODEL)]
    model: String,
    /// Specify once or twice to set log level info or debug repsectively.
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
    /// Face direction
    #[clap(arg_enum, short, long, default_value = "auto")]
    direction: FaceDirection,
    /// Save preprocessed image.
    #[clap(long)]
    preprocessed: Option<String>,
    /// Save raw output (3D array with shape [6, H, W]).
    #[clap(long)]
    raw: Option<String>,
    /// Save heatmap image.
    #[clap(long)]
    heatmap: Option<String>,
    /// Save anatomical point coordinates in json.
    #[clap(long)]
    json: Option<String>,
    /// No background (input image) for the output.
    #[clap(short, long)]
    no_background: bool,
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

    let try_dicom = if let Some(ext) = Path::new(&args.input).extension() {
        info!("Input image extension {:?}", ext);
        ext == "dcm" || ext == "dicom"
    } else {
        false
    };

    info!("Open {}", args.input);
    let original_img = if try_dicom {
        info!("Open as dicom");
        let dicom_img = c2c7demo::load_dicom(&args.input);
        match dicom_img {
            Ok(img) => img,
            Err(e) => {
                debug!("{:?}", e.to_string());
                image::open(&args.input).unwrap()
            }
        }
    } else {
        image::open(&args.input).unwrap()
    };
    let target_height = c2c7demo::TARGET_HEIGHT as u32;
    let (width, height) = original_img.dimensions();
    let gray_img = original_img.grayscale();

    let gray_img = match args.direction {
        FaceDirection::Left => gray_img,
        FaceDirection::Right => {
            info!("Flip input image");
            match gray_img {
                DynamicImage::ImageLuma8(i) => {
                    DynamicImage::ImageLuma8(image::imageops::flip_horizontal(&i))
                }
                DynamicImage::ImageLuma16(i) => {
                    DynamicImage::ImageLuma16(image::imageops::flip_horizontal(&i))
                }
                _ => panic!("Invalid image format"),
            }
        }
        FaceDirection::Auto => gray_img,
    };
    let new_width: u32 = ((width as f64) * (target_height as f64) / (height as f64)).round() as u32;
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

    if let Some(filename) = args.preprocessed {
        info!("Save preprocessed image {}", filename);
        img.save(filename).unwrap();
    }

    let pad_width = ((new_width as f64 / 256.0).ceil() * 256.0) as u32;
    let (input_width, input_height) = (pad_width, target_height);
    let batch_size = match args.direction {
        FaceDirection::Auto => 2,
        _ => 1,
    };

    let model_path = c2c7demo::resolve_model_path(&args.model)?;

    info!("Load model");
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec!(batch_size, 1, input_height as usize, input_width as usize),
            ),
        )?
        .into_optimized()?
        .into_runnable()?;

    let input_tensor: Tensor = match args.direction {
        FaceDirection::Auto => {
            info!("Face direction Auto");
            c2c7demo::create_input_batch(&img, input_width, input_height).into()
        }
        _ => ndarray::Array4::from_shape_fn(
            (1, 1, input_height as usize, input_width as usize),
            |(_, c, y, x)| {
                if img.in_bounds(x as _, y as _) {
                    img[(x as _, y as _)][c] as f32 / 255.0
                } else {
                    0.0
                }
            },
        )
        .into(),
    };

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
    let best_batch: usize = match args.direction {
        FaceDirection::Auto => c2c7demo::choose_best_batch(&arr4) as _,
        _ => 0,
    };
    let flip_needed = best_batch != 0;

    let arr3 = arr4
        .slice(ndarray::s![
            best_batch,
            ..,
            0..img_height as usize,
            0..img_width as usize
        ])
        .map(|v| (v * 255.0) as u8);

    if let Some(filename) = args.raw {
        info!("Save raw output with shape {:?} {}", arr3.shape(), filename);
        std::fs::write(&filename, arr3.as_slice().unwrap()).unwrap();
    }

    if let Some(filename) = args.heatmap {
        info!("Save heatmap {}", filename);
        c2c7demo::extract_heatmap(&arr3)?.save(filename).unwrap();
    }

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
    if let Some(filename) = args.json {
        let scale = original_img.dimensions().1 as f32 / c2c7demo::TARGET_HEIGHT as f32;
        let scaled_points: Vec<c2c7demo::Point> = optimal_points
            .into_iter()
            .map(|(x, y)| (scale * x, scale * y))
            .collect();
        let scaled_data =
            PointData::new(&scaled_points, &labels, img_width as _, img_height as _, "");
        info!("Save json {}", filename);
        scaled_data.save(&filename).unwrap();
    };

    let background = if args.no_background {
        None
    } else {
        match gray_img {
            DynamicImage::ImageLuma8(img) => Some(DynamicImage::ImageLuma8(img)),
            DynamicImage::ImageLuma16(img) => Some(c2c7demo::luma16toluma8(&img)),
            _ => None,
        }
    };
    let svg_output = args.output.ends_with(".svg");
    let document = draw(data, background, !svg_output);
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
