import './App.css';
import { useEffect, useState } from 'react'
import Dropzone from 'react-dropzone'
import init, { greet } from "c2c7demo";

function App() {
  useEffect(() => {
    init().then(() => {
      console.log("wasm loaded.")
    })
  }, [])
  const [inputImage, setInputImage] = useState(new Blob());
  const [resultImage, setResultImage] = useState("");
  const processImage = async (filename: any) => {
    const imageUrl = URL.createObjectURL(filename);
    const resp = await fetch(imageUrl);
    setInputImage(await resp.blob());
  }
  return (
    <div className="App">
      {
        inputImage.size ?
          <div className="main">
            <div>
              <h3>Input Image</h3>
              <img src={URL.createObjectURL(inputImage)} />
            </div>
            {resultImage.length ?
              <>
                <div>
                  <h3>Result</h3>
                  <div className="overlay">
                    <img className="bg" src={URL.createObjectURL(inputImage)} />
                    <img className="fg" src="img/overlay.svg" />
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
                  <button title="Restart from the start" onClick={() => {setInputImage(new Blob())}}>Reset</button>
                </div>
              </>
              :
              <div className="buttons">
                <button title="Let the AI do its job."
                  onClick={(event) => { console.log("button pressed"); setResultImage("yes") }}>Measure
                </button>
              </div>
            }

          </div>
          :
          <Dropzone onDrop={acceptedFiles => { console.log(acceptedFiles); processImage(acceptedFiles[0]) }}>
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
