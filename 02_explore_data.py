import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


def plot_image(x):
    """Plot an image

    Parameters
    ----------
    x : numpy.ndarray
        From cv2.imread()
        
    Notes
    -----
    https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748
    """
    b, g, r = cv2.split(x)
    img = cv2.merge([r, g, b])
    plt.imshow(img)


def plot_bounding_box(img, img_name, img_details):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i in img_details.query('image_name == @img_name').index:
        xmin = img_details.loc[i, 'xmin']
        xmax = img_details.loc[i, 'xmax']
        ymin = img_details.loc[i, 'ymin']
        ymax = img_details.loc[i, 'ymax']
        rect =  patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    


training_annotations = pd.read_csv('data/trainingset_annotations.csv')
training_annotations.columns = [i.strip() for i in training_annotations.columns]

img_name = training_annotations.sample(1)['image_name'].values[0]
img_path = f'data/trainingset/{img_name}'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_details = training_annotations.query('image_name == @img_name')
xmin = img_details['xmin']

plot_bounding_box(img, img_name, training_annotations)
