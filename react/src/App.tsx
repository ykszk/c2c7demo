import './App.css';
import { useEffect, useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import init, { decode_image, create_input_tensor, process_output, calc_tensor_width } from "c2c7demo";
import * as ort from 'onnxruntime-web';

class ImageFile {
  filename: string;
  b64: string;
  width: number;
  height: number;
  arr?: Uint8Array;

  constructor(filename: string, b64: string, width: number, height: number, arr?: Uint8Array) {
    this.filename = filename;
    this.b64 = b64;
    this.width = width;
    this.height = height;
    this.arr = arr;
  }
  static default(): ImageFile {
    return new ImageFile("", "", 0, 0);
  }
  is_default(): boolean {
    return this.filename === "" && this.b64 === "";
  }
}

class Result {
  image: string;
  inferenceTime: number;

  constructor(image: string, inferenceTimer: number) {
    this.image = image;
    this.inferenceTime = inferenceTimer;
  }
}

// cf. [Classify images in a web application with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html)
async function runInference(session: ort.InferenceSession, preprocessedData: any): Promise<[Float32Array, number]> {
  // Get start time to calculate inference time.
  const start = new Date();
  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  // Run the session inference.
  const outputData = await session.run(feeds);
  // Get the end time to calculate inference time.
  const end = new Date();
  // Convert to seconds.
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];
  const arr = output.data as Float32Array;
  return [arr, inferenceTime];
}

function downloadAsSVG(text: string, name: string) {
  const blob = new Blob([text], { type: 'image/svg+xml' });
  var link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  link.remove();
}

function App() {
  useEffect(() => {
    init().then(() => {
      console.log("wasm loaded.")
    })
  }, [])

  const onDrop = useCallback(acceptedFiles => {
    processImage(acceptedFiles[0]);
  }, [])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop })
  const [inputImage, setInputImage] = useState(ImageFile.default());
  const [errMessage, setErrMessage] = useState("");
  const [onnxModel, setOnnxModel] = useState(new ArrayBuffer(0));
  const [measureDisabled, setMeasureDisabled] = useState(false);
  const [result, setResult] = useState(new Result("", 0));
  // const [fgSrc, setFgSrc] = useState("");
  const processImage = async (file: File) => {
    const filename = file.name;
    console.log(filename);
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = function () {
      const buf = reader.result as ArrayBuffer;
      const arr = new Uint8Array(buf);
      try {
        const imgb64 = decode_image(arr);
        setInputImage(new ImageFile(filename, imgb64.b64, imgb64.width, imgb64.height, arr));
      } catch (error) {
        console.error(error);
        setErrMessage(error as string);
        setInputImage(ImageFile.default());
      }
    }
  }
  const blob = new Blob([result.image], { type: 'image/svg+xml' });
  const fgURL = URL.createObjectURL(blob);
  const reset_button = (
    <button title="Restart from the start" onClick={() => { setInputImage(ImageFile.default()); setResult(new Result("", 0)); setErrMessage(""); setMeasureDisabled(false) }}>Reset</button>
  )

  function run_model(model: ArrayBuffer) {
    console.log('Create onnx session');
    ort.InferenceSession.create(
      model,
      {
        executionProviders: ["wasm", "webgl"], graphOptimizationLevel: 'all'
      }
    ).then(session => {
      const tensor_raw = create_input_tensor(inputImage.arr as Uint8Array);
      const tensor_width = calc_tensor_width(inputImage.width, inputImage.height);
      const dims = [2, 1, 768, tensor_width];
      console.log(inputImage.width, dims);
      const input_tensor = new ort.Tensor("float32", tensor_raw, dims)
      console.log('Run inference');
      runInference(session, input_tensor).then(v => {
        const [results, inferenceTime] = v;
        console.log(results.length, inferenceTime);
        const svg_str = process_output(inputImage.arr as Uint8Array, results, tensor_width, inputImage.width, inputImage.height);
        setResult(new Result(svg_str, inferenceTime));
      }).catch(reason => { console.error(reason); setErrMessage("Failed to run the inference.") });
    }
    ).catch(reason => { console.error(reason); setErrMessage("Failed to create a onnx session.") })
  }
  if (errMessage.length !== 0) {
    return (
      <div className="App">
        <p className="error">{errMessage}</p>
        {reset_button}
      </div>
    )
  } else if (inputImage.is_default()) {
    return (
      <div className="App">
        <div className={"dropzone" + (isDragActive ? " dragActive" : "")} {...getRootProps()}>
          <input {...getInputProps()} />
          {
            isDragActive ?
              <p>Open the file ...</p> :
              <p>Drop lateral radiograph of cervical spine, or click to choose. PNG, JPEG or DICOM (experimental).</p>
          }
        </div>
      </div>
    )
  }
  return (
    <div className="App">
      <div className="main">
        <div>
          <h3>Input Image</h3>
          <img alt="input" src={inputImage.b64} />
        </div>
        {result.image.length ?
          <>
            <div>
              <h3>Result</h3>
              <img alt="result" src={fgURL} />
            </div>
            <div className="buttons">
              <p className="smallNote">Inference took {result.inferenceTime.toFixed(1)} secs.</p>
              <button title="Save output" onClick={() => { downloadAsSVG(result.image, (inputImage.filename.substring(0, inputImage.filename.lastIndexOf('.')) || inputImage.filename) + ".svg") }}>Save</button>
              {reset_button}
            </div>
          </>
          :
          <div className="buttons">
            <button title="Let the AI do some job." disabled={measureDisabled}
              onClick={(event) => {
                try {
                  setMeasureDisabled(true);
                  if (onnxModel.byteLength === 0) {
                    console.log("Fetch onnx model")
                    fetch(process.env.PUBLIC_URL + "/c2c7_MNetV2.onnx").then(
                      response => {
                        response.arrayBuffer().then(buf => { setOnnxModel(buf); run_model(buf); });
                      }
                    );
                  } else {
                    // setTimeout is used to re-render this component with "measure" button disabled. There maybe some better way to force render the component.
                    setTimeout(() => { run_model(onnxModel); }, 10);
                  }
                } catch (error) {
                  console.error(error);
                  setErrMessage(error as string);
                }

              }}> {measureDisabled ? "Please wait. Measuring takes about 5 - 30 seconds." : "Measure"}
            </button>
            {measureDisabled ? <></> : reset_button}
          </div>
        }
      </div>
    </div >
  );
}

export default App;
