# Overview
This package contains a python program that demonstrates automated C2-C7 Cobb angle measuring.
The program is tested on Windows 10 and Ubuntu 20.04.

Unlike the web application and executables, this python version uses GPU when it is available.

# System Requirements

Python 3 is required to run the program.

## Dependencies

The main dependencies of the program are the following python packages.
```
torch==1.10.1
numpy==1.21.0
scipy==1.7.3
pydicom==2.2.2
segmentation-models-pytorch==0.2.1
```

The software may or may not work with other package versions.
Full list of the dependencies is available in `requirements.txt`.

# Installation Guide

It takes about five minutes to install the dependencies.
Replace `python` and `pip` commands with `python3` and `pip3` respectively if needed.

## (Optional) Setup a temporary environment
Create an environment.
``` shell
python -m venv c2c7
```

Activate the environment.  
(Windows)
``` cmd
c2c7\Scripts\activate
```

(Linux)
``` shell
source c2c7/bin/activate
```

## Install dependencies
``` shell
pip install -r requirements.txt
```

## Download weights
Download `c2c7_ENetB4.pth` from [google drive](https://drive.google.com/drive/folders/18Jet4hS7PALKxHSdak3nSURKlO06CLSU?usp=sharing) and save it as `weights/c2c7_ENetB4.pth`.

# Run
Run the program in a terminal as follows.
``` shell
python c2c7_cobb.py <input> <output>
```
, where `<input>` is the input x-ray image and `<output>` is the output image.

`<input>` image can be a DICOM image whereas `<output>` cannnot.