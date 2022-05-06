# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run build:wasm`

Builds wasm

# Build

## Prepare wasm for development and deployment
Copy onnxruntime wasm as follows for developping.  
`cd react && mkdir -p public/static/js && cp -r node_modules/onnxruntime-web/dist/*.wasm public/static/js`

## Download trained model
`cd react && curl -L "https://drive.google.com/uc?export=download&id=1qUwbOi1Gx_t2BvXm9x-qhYEoHWU4SoXP" -o public/c2c7_MNetV2.onnx`