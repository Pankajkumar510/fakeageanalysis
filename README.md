6# Age Estimation Model — UTKFace + ResNet18

## 1. Dataset
Download UTKFace from:
https://www.kaggle.com/datasets/jangedoo/utkface-new

Extract the images into:
data/UTKFace/

The filenames should look like:
25_1_2_20170116174525125.jpg

The first number is the age.

## 2. Install
pip install -r requirements.txt

## 3. Train
python train_age.py

The trained model will be saved as:
models/age_model.pth

## 4. Test
Put a face image at:
test.jpg

Run:
python predict_age.py

Or:
python predict_age.py --image myface.jpg --model models/age_model.pth

## Important
This is an age-estimation model, not an exact-age identification system.
For video use, the next step is face detection + frame tracking + age prediction.
