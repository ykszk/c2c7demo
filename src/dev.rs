use imageproc::region_labelling::{connected_components, Connectivity};
use std::fs::{self, File};
use std::io::Read;
use tract_onnx::prelude::tract_ndarray::{self, s, Axis};
#[macro_use]
extern crate log;
use env_logger::{Builder, Env};

type Point = (f32, f32);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = Env::default().filter_or("LOG_LEVEL", "debug");
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Seconds))
        .init();
    let filename = "result.bin";
    let (chs, height, width) = (6, 768, 768);
    let mut f = File::open(&filename).expect("no file found");
    let metadata = fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    if buffer.len() != chs * height * width {
        panic!("Invalid input size")
    } else {
        debug!("{} loaded.", filename);
    }
    let arr = tract_ndarray::Array3::from_shape_vec((chs, height, width), buffer)?;
    debug!("array shape: {:?}", arr.shape());
    let ch_axis = Axis(0);
    let mut candidates: Vec<Vec<Point>> = Vec::new();
    for (i_ch, img_ch) in arr.axis_iter(ch_axis).enumerate() {
        if i_ch > 3 {
            // skip affinity maps
            break;
        }
        let max_value = img_ch.iter().max().unwrap();
        let thresh = max_value / 2;
        debug!(
            "Max value for ch {} is {}. thresh is {}",
            i_ch, max_value, thresh
        );
        let bin_arr = img_ch.mapv(|v| if v > thresh { 1u8 } else { 0u8 });
        let bin_img =
            image::GrayImage::from_raw(width as _, height as _, bin_arr.as_slice().unwrap().into())
                .unwrap();
        let cced = connected_components(&bin_img, Connectivity::Four, image::Luma([0u8]));
        let n_cc = *cced.iter().max().unwrap();
        assert!(n_cc > 0);
        debug!("# of CC is {}", n_cc);

        let mut cand_points: Vec<Point> = Vec::new();
        for cc_val in 1..(n_cc + 1) {
            let cc_max_value = img_ch
                .indexed_iter()
                .filter_map(|((y, x), val)| {
                    if cced.get_pixel(x as _, y as _)[0] == cc_val {
                        Some(*val)
                    } else {
                        None
                    }
                })
                .max()
                .unwrap();
            let mut argmax: Vec<(usize, usize)> = Vec::new();
            for ((x, y), val) in img_ch.indexed_iter() {
                if *val == cc_max_value {
                    argmax.push((x, y));
                }
            }
            assert!(
                argmax.len() > 0,
                "No argmax (coordinates with maximum value) found."
            );
            let sum_argmax: (usize, usize) = argmax
                .iter()
                .fold((0, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
            let mean_argmax = (
                sum_argmax.0 as f32 / argmax.len() as f32,
                sum_argmax.1 as f32 / argmax.len() as f32,
            );
            cand_points.push(mean_argmax);
        }
        assert!(cand_points.len() > 0);
        candidates.push(cand_points);
    }
    assert_eq!(candidates.len(), 4);
    debug!("Candidates {:?}", candidates);

    let mut optimal_pairs: Vec<(Point, Point)> = Vec::new();
    for c2c7 in 0..2 {
        let cand_left = &candidates[c2c7 * 2];
        let cand_right = &candidates[c2c7 * 2 + 1];
        if cand_left.len() == 1 && cand_right.len() == 1 {
            debug!("No need to find optimal point pairs");
            optimal_pairs.push((cand_left[0], cand_right[0]));
        } else {
            debug!("Find optimal point pairs");
            let affinity = arr.slice(s![c2c7 + 4, .., ..]);
            let mut scores: Vec<(usize, Point, Point)> =
                Vec::with_capacity(cand_left.len() * cand_right.len());
            for left in cand_left.iter() {
                for right in cand_right.iter() {
                    let l = (left.0 as isize, left.1 as isize);
                    let r = (right.0 as isize, right.1 as isize);
                    let score: usize = line_drawing::Bresenham::new(l, r)
                        .map(|(x, y)| affinity[[y as usize, x as usize]] as usize)
                        .sum();
                    scores.push((score, *left, *right));
                }
            }
            let (_optimal_score, left, right) = scores.into_iter().max_by_key(|t| t.0).unwrap();
            optimal_pairs.push((left, right));
        }
    }
    debug!("Optimal points\n{:?}", optimal_pairs);

    return Ok(());
}
