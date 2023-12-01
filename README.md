# Group Project: Facial Expression Recognition

> Group members:
> 
> Qingkuo Li 
> 
> Xiaotong Sun

## Project Environment
<img src=https://img.shields.io/badge/Python-%3E%3D3.7-blue> <img src=https://img.shields.io/badge/TensorFlow-%3E%3D2.9.2-blue> <img src=https://img.shields.io/badge/Keras-%3E%3D2.3.1-blue>

To reproduce the project codes, it is recommended to create a virtual environment using Anaconda (or any other environment manager):
```bash
conda create -n tensorflow python=3.7
```
and activate the environment:
```bash
conda activate tensorflow
```
Then, a quick way to install all of required packages is using `requirements.txt` file provided in the project repo:
```bash
pip install requirements.txt
```
Or install them manually (with `conda install` or `pip install`):
- numpy>=1.21.5
- pandas>=1.3.5
- scipy>=1.7.3
- matplotlib>=3.5.3
- tensorflow>=2.9.2
- keras>=2.3.1
- scikit-learn>=1.0.2
- opencv-python>=4.6.0
- opencv-contrib-python>=4.6.0
- argparse>=1.4.0
- Augmentor>=0.2.10

## Usage
### 1. Data Pre-processing
Clone or download this repository under the environment, decompress the `datasets.zip` file, and make sure that the project codes is organised in the following directory structure:
- ML_GroupProject/
  - csv2img.py
  - faceprocess.py
  - img2csv.py
  - imgAug.py
  - train_cnn.py
  - train_cnn.ipynb
  - train_seq.py
  - train_seq.ipynb
  - datasets/
    - 0/
    - 1/
    - 2/
    - ...

To convert raw datasets into greyscale images with the size of 48*48, run the `faceprocess.py` script, it will process all of images in the given directory and save them in a new folder `/datasets/processed`:
```bash
python faceprocess.py -p ./datasets
```
To randomly augment the processed dataset, run the `imgAug.py` script:
```bash
python imgAug.py -p ./datasets/processed
```
To convert all of images into one single CSV (comma-separated values) file for future training, run the `img2csv.py` script:
```bash
python img2csv.py -p ./datasets/processed
```
Here, for the convenience of training, a converted CSV file `dataset.csv` is also provided in the repository.

### 2. Training
To train the CNN model, run the Python script `train_cnn.py`:
```bash
python train_cnn.py
```
Or if using Jupyter, a Jupyter version `train_cnn.ipynb` is also provided.

To train the sequential models, run the Python script `train_seq.py`:
```bash
python train_seq.py
```
Or if using Jupyter, a Jupyter version `train_seq.ipynb` is also provided.

### 3. Real-time Facial Expression Recognition
Clone or donwload this repository, put the trained model into `model` folder, and ensure that the project codes is organised in the following directory structure:
- ML_GroupProject/
  - detector/
    - haarcascade_frontalface_alt.xml
  - model/
    - facial_model.h5
  - camera.py
  
And just run the script `camera.py`:
```bash
python camera.py
```
then it will be able to recognize facial expressions in real-time using camera (press `Q` to exit).

Here, a pre-trained model `facial_model.h5` is provided in the repository for recognition.

## Results
We used a Tesla T4 NVIDIA GPU for training, a full training session took 21mins 50secs. The performance is as follows (accuracy on the test set):
| Model | Accuracy |
| ----- | -------- |
| CNN   | 0.975    |
| KNN   | 0.370    |
| SVM   | 0.495    |

## Contributions
- Data set collection: Qingkuo Li, Xueni Fu
- Codes:
  - csv2img.py: Qingkuo Li
  - faceprocess.py: Qingkuo Li
  - img2csv.py: Qingkuo Li
  - imgAug.py: Qingkuo Li
  - train_cnn.py, train_cnn.ipynb: Qingkuo Li, Xueni Fu
  - train_seq.py, train_seq.ipynb: Xueni Fu
  - camera.py: Qingkuo Li
- Report: Qingkuo Li, Xueni Fu
