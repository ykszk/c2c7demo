use assert_cmd::prelude::*;
use std::process::Command;

#[test]
fn file_doesnt_exist() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("c2c7angle")?;

    let json_filename = "tests/test_output.json";
    cmd.arg("tests/img/extension.jpg")
        .arg("tests/img/test_output.svg")
        .arg("--json")
        .arg(json_filename);
    cmd.assert().success();
    let file = std::fs::File::open(json_filename)?;
    let reader = std::io::BufReader::new(file);

    let json_data: c2c7demo::PointData = serde_json::from_reader(reader)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    assert_eq!(json_data.shapes.len(), 4);
    let true_points = [
        [502.08, 685.41],
        [610.41, 637.5],
        [785.763, 1197.56],
        [887.499, 1129.16],
    ];
    for (shape, true_point) in json_data.shapes.into_iter().zip(true_points) {
        let point = shape.points[0];
        assert_eq!((point.0 / 10.0) as usize, (true_point[0] / 10.0) as usize);
        assert_eq!((point.1 / 10.0) as usize, (true_point[1] / 10.0) as usize);
    }
    Ok(())
}
