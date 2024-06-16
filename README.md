# DarEM

## Installation

### Linux
Install the following prerequisites

 * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - Required for training or fast prediction on a GPU. Not necessary for slow prediction on a CPU
 * [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
    - Using other ways to create virtual python environements is possible, but our installation instructions are based on conda
 * gcc
    - Run `sudo apt install gcc` in a terminal
 * [Protobuf V3.19](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.4)

Open a terminal and run
```bash
conda create -n DarEM python==3.9 cudnn
conda activate DarEM
git clone https://github.com/tensorflow/models.git
git checkout 457bcb8595903331932e2faf95bec8ba69e04688
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
cd ../../
git clone https://github.com/DavidKleindienst/DarEM
cd DarEM
pip install -r requirements.txt
```
To run DarEM run on `python Train.py` for training or `python Predict.py` for Prediction.


### Windows
Ensure the following prerequisites are installed:
 * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - Required for training or fast prediction on a GPU. Not necessary for slow prediction on a CPU
 * [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
    - Using other ways to create virtual python environements is possible, but our installation instructions are based on conda
 * [Protobuf V3.19](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.4)
    - Unzip the downloaded file and add the path to /bin to your PATH environment variable
 * [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    - In the installer, select "Desktop Development with C++".
    - A reboot may be required

Open a new powershell and run
```powershell
conda create -n DarEM python==3.9 cudnn
conda activate DarEM
git clone https://github.com/tensorflow/models.git
git checkout 457bcb8595903331932e2faf95bec8ba69e04688
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
cd ../../
git clone https://github.com/DavidKleindienst/DarEM
cd DarEM
pip install -r requirements.txt
```

To run DarEM doubleclick on Train.cmd for training or Predict.cmd for Prediction.

## Citation
If you've used this software for your research, please cite

David Kleindienst, Tommaso Costanzo, Ryuichi Shigemoto.
Automated Imaging and Analysis of Synapses in Freeze-Fracture Replica Samples with Deep Learning.
*Neuromethods, Vol. 212*, Joachim Lübke and Astrid Rollenhagen (Eds)
in Press

...
