# https://appen.com/datasets/parking-sign-detection/

import requests
import pandas as pd

# training

url_training_imgs = 'https://datasets.appen.com/appen_datasets/ParkingSign-StreetView/trainingset.zip'
r = requests.get(url_training_imgs, allow_redirects=True)
open('data/trainingset_images.zip', 'wb').write(r.content)

url_training_annotations = 'https://datasets.appen.com/appen_datasets/ParkingSign-StreetView/trainingset_annotations.csv'
trainingset_annotations = pd.read_csv(url_training_annotations)
trainingset_annotations.to_csv('data/trainingset_annotations.csv', index=False)

# validation

url_valid_imgs = 'https://datasets.appen.com/appen_datasets/ParkingSign-StreetView/validationset.zip'
r = requests.get(url_valid_imgs, allow_redirects=True)
open('data/validset_images.zip', 'wb').write(r.content)

url_valid_annotations = 'https://datasets.appen.com/appen_datasets/ParkingSign-StreetView/validationset_annotations.csv'
validset_annotations = pd.read_csv(url_valid_annotations)
validset_annotations.to_csv('data/validset_annotations.csv', index=False)