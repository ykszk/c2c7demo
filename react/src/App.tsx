import './App.css';
import { useEffect, useState } from 'react'
import Dropzone from 'react-dropzone'
import init, { decode_image, preprocess_image, create_input_tensor, process_output } from "c2c7demo";
// import { Tensor, InferenceSession } from "onnxruntime-web";
import * as ort from 'onnxruntime-web';

const SVG_HEAD = `<?xml version="1.0" encoding="UTF-8"?>\n
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">`;

class ImageFile {
  filename: string;
  b64: string;
  arr?: Uint8Array;

  constructor(filename: string, b64: string, arr?: Uint8Array) {
    this.filename = filename;
    this.b64 = b64;
    this.arr = arr;
  }
  static default(): ImageFile {
    return new ImageFile("", "");
  }
  is_default(): Boolean {
    return this.filename === "" && this.b64 === "";
  }
}

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
  const inferenceTime = (end.getTime() - start.getTime())/1000;
  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];
  console.log(output.dims, output.type);
  const arr = output.data as Float32Array;
  //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  // var outputSoftmax = ort.softmax(Array.prototype.slice.call(output.data));
  //Get the top 5 results.
  // var results = imagenetClassesTopK(outputSoftmax, 5);
  // console.log('results: ', results);
  // return [results, inferenceTime];
  return [arr, inferenceTime];
}

function App() {
  useEffect(() => {
    init().then(() => {
      console.log("wasm loaded.")
    })
  }, [])
  const [inputImage, setInputImage] = useState(ImageFile.default());
  const [resultImage, setResultImage] = useState("");
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
        const b64 = decode_image(arr, filename);
        setInputImage(new ImageFile(filename, b64, arr));
      } catch (error) {
        console.error(error);
        setInputImage(ImageFile.default());
      }
    }
  }
  let blob = new Blob([fgSrc], {type: 'image/svg+xml'});
  let fgURL = URL.createObjectURL(blob);
  return (
    <div className="App">
      {
        !inputImage.is_default() ?
          <div className="main">
            <div>
              <h3>Input Image</h3>
              <img alt="input" src={inputImage.b64} />
            </div>
            {resultImage.length ?
              <>
                <div>
                  <h3>Result</h3>
                  <div className="overlay">
                    <img alt="background" className="bg" src={inputImage.b64} />
                    {/* {fgSrc} */}
                    <img alt="foreground" className="fg" src={fgURL} />
                  </div>
                </div>
                <div className="buttons">

                  <label>
                    <input type="checkbox" />
                    Lines
                  </label>
                  <label>
                    <input type="checkbox" />
                    Points
                  </label>
                  <label>
                    <input type="checkbox" />
                    Heatmap
                  </label>
                  
                  <button title="Save output">Save</button>
                  <button title="Restart from the start" onClick={() => { setInputImage(ImageFile.default()); setResultImage("") }}>Reset</button>
                </div>
              </>
              :
              <div className="buttons">
                <button title="Let the AI do its job."
                  onClick={async (event) => {
                    console.log("Preprocess");


                    // const mid_img = preprocess_image(inputImage.arr as Uint8Array, inputImage.filename);
                    // inputImage.b64 = mid_img;
                    console.log("Create onnx session")
                    const session = await ort.InferenceSession.create(
                      "c2c7.onnx",
                      {
                        executionProviders: ["webgl"], graphOptimizationLevel: 'all'
                      }
                    );
                    // const tensor_raw = 
                    // console.log("Load model");
                    // fetch("c2c7.onnx").then(async response => {
                    //   console.log("Model loaded")
                    //   console.log(response);
                    //   const model_u8 = new Uint8Array(await response.arrayBuffer());
                    //   console.log(model_u8.length);
                    //   console.log('Run inference');
                    //   const startTime = Date.now();
                    //   const svg_str = apply_model(inputImage.arr as Uint8Array, inputImage.filename, model_u8);
                    //   const endTime = Date.now(); 
                    //   console.log("inference took ", (endTime - startTime)/1000/60, " mins");
                    //   console.log(svg_str);
                    // })
                    const tensor_raw = create_input_tensor(inputImage.arr as Uint8Array, inputImage.filename);
                    const dims = [2, 1, 768, 768];
                    const input_tensor = new ort.Tensor("float32", tensor_raw, dims)
                    console.log('Run inference');
                    const [results, inferenceTime] = await runInference(session, input_tensor);
                    console.log(results.length, inferenceTime);
                    const svg_str = process_output(results);
                    console.log(svg_str);
                    setFgSrc(svg_str)
                    // setFgSrc("data:image/svg+xml;base64," + btoa(SVG_HEAD + svg_str));
                    // setInputImage(inputImage)
                    setResultImage("yes")
                  }}>Measure
                </button>
              </div>
            }

          </div>
          :
          <Dropzone onDrop={acceptedFiles => { processImage(acceptedFiles[0]) }}>
            {({ getRootProps, getInputProps }) => (
              <section>
                <div className="dropzone" {...getRootProps()}>
                  <input {...getInputProps()} />
                  <p>Drag input image, or click to choose a file.</p>
                </div>
              </section>
            )}
          </Dropzone>
      }
    </div >
  );
}

export default App;
