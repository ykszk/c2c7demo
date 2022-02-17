import './App.css';
import { useEffect, useState } from 'react'
import Dropzone from 'react-dropzone'
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
  console.log(output.dims, output.type);
  const arr = output.data as Float32Array;
  return [arr, inferenceTime];
}

function App() {
  useEffect(() => {
    init().then(() => {
      console.log("wasm loaded.")
    })
  }, [])
  const [inputImage, setInputImage] = useState(ImageFile.default());
  const [errMessage, setErrMessage] = useState("");
  const [measureDisabled, setMeasureDisabled] = useState(false);
  const [fgSrc, setFgSrc] = useState("");
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
  const blob = new Blob([fgSrc], { type: 'image/svg+xml' });
  const fgURL = URL.createObjectURL(blob);
  const reset_button = (
    <button title="Restart from the start" onClick={() => { setInputImage(ImageFile.default()); setFgSrc(""); setErrMessage(""); setMeasureDisabled(false) }}>Reset</button>
  )
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
        <Dropzone onDrop={async acceptedFiles => processImage(acceptedFiles[0])}>
          {({ getRootProps, getInputProps }) => (
            <section>
              <div className="dropzone" {...getRootProps()}>
                <input {...getInputProps()} />
                <p>Drag input image, or click to choose a file. PNG, JPEG or DICOM (experimental).</p>
              </div>
            </section>
          )}
        </Dropzone>
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
        {fgSrc.length ?
          <>
            <div>
              <h3>Result</h3>
              <div className="overlay">
                <img alt="background" className="bg" src={inputImage.b64} />
                <img alt="foreground" className="fg" src={fgURL} />
              </div>
            </div>
            <div className="buttons">
              <button title="Save output">Save</button>
              {reset_button}
            </div>
          </>
          :
          <div className="buttons">
            <button title="Let the AI do some job." disabled={measureDisabled}
              onClick={(event) => {
                try {
                  setMeasureDisabled(true);
                  console.log("Create onnx session")
                  ort.InferenceSession.create(
                    "c2c7.onnx",
                    {
                      executionProviders: ["webgl"], graphOptimizationLevel: 'all'
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
                      const svg_str = process_output(results, tensor_width, inputImage.width, inputImage.height);
                      setFgSrc(svg_str)
                    }).catch(reason => { console.error(reason); setErrMessage("Failed to run the inference.") });
                  }
                  ).catch(reason => { console.error(reason); setErrMessage("Failed to create a onnx session.") })
                } catch (error) {
                  console.error(error);
                  setErrMessage(error as string);
                }

              }}> {measureDisabled ? "Please wait. Measuring takes about 5 - 30 seconds." : "Measure"}
            </button>
            {measureDisabled ? "" : reset_button}
          </div>
        }
      </div>
    </div >
  );
}

export default App;
