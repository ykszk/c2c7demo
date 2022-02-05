use std::error::Error;
use std::path::Path;

#[macro_use]
extern crate log;
use clahe::clahe;
use clap::Parser;
use dicom::dictionary_std;
use dicom::object::open_file;
use env_logger::{Builder, Env};
use image::error::{ImageFormatHint, UnsupportedError};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input: String,
    /// Output image filename
    output: String,

    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
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
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();

    let try_dicom = if let Some(ext) = Path::new(&args.input).extension() {
        info!("Input image extension {:?}", ext);
        ext == "dcm" || ext == "dicom"
    } else {
        false
    };

    info!("Open {}", args.input);
    let img = if try_dicom {
        info!("Open as dicom");
        let obj = open_file(&args.input);
        if let Ok(obj) = obj {
            let pixel_representation: u8 = obj.element_by_name("PixelRepresentation")?.to_int()?;
            let bits_allocated: u8 = obj.element_by_name("BitsAllocated")?.to_int()?;
            let bits_stored: u8 = obj.element_by_name("BitsStored")?.to_int()?;
            debug!("pixel representation {}, bits_stored {}", pixel_representation, bits_stored);
            // let compression: u8 = obj.element_by_name("LossyImageCompression")?.to_int()?;
            assert_eq!(bits_allocated, 16, "Not a 16bit dicom");
            let width: u32 = obj.element(dictionary_std::tags::COLUMNS)?.to_int()?;
            let height: u32 = obj.element(dictionary_std::tags::ROWS)?.to_int()?;

            let pixel_data_bytes = obj.element(dicom::core::Tag(0x7FE0, 0x0010))?.to_bytes()?;

            type PixelType = u16;
            let pixels: Vec<PixelType> =
                unsafe { (pixel_data_bytes.into_owned().align_to::<PixelType>().1).to_vec() };
            let pixels_u16: Vec<u16> = pixels.into_iter().collect();
            type OutputPixelType = u16;
            let buf: ImageBuffer<image::Luma<OutputPixelType>, Vec<OutputPixelType>> =
                ImageBuffer::from_raw(width, height, pixels_u16).unwrap();
            DynamicImage::ImageLuma16(buf)
        } else {
            info!("Failed to open as a dicom. Falling back to regular format.");
            image::open(&args.input).unwrap()
        }
    } else {
        image::open(&args.input).unwrap()
    };
    let target_height = 768;
    let (width, height) = img.dimensions();
    let img = img.grayscale();
    let new_width: u32 = ((width as f64) * (target_height as f64) / (height as f64)).round() as u32;
    debug!("Original image size: {} {}", width, height);
    info!("Resize to w{} h{}", new_width, target_height);
    let img = img.resize_exact(new_width, target_height, FilterType::Triangle);
    let img = match img {
        DynamicImage::ImageLuma8(img) => {
            debug!("input u8 to clahe");
            clahe(&img, 16, 16, 10)?}
            ,
        DynamicImage::ImageLuma16(img) => {
            debug!("input u16 to clahe");
            clahe(&img, 16, 16, 10)?
        },
        _ => {
            let hint =
                ImageFormatHint::Name("u8 or u16 image is expected".to_string());
            return Err(UnsupportedError::from(hint).into());
        }
    };
    info!("Save {}", args.output);
    img.save(&args.output).unwrap();
    Ok(())
}
