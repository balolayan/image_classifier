
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image
import time
import json

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/wild_pansy.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./1593454200.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_save = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

    
def load_checkpoint(model_save):
    
    reloaded = tf.keras.models.load_model(model_save, custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (224, 224))/255
    image = image.numpy()
    return image


    
def predict(image_path, model, topk):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, topk)
    print("These are the top propabilities",top_values.numpy()[0])
    top_classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    
    return top_values.numpy()[0], top_classes

def Sanity_Checkpoint(model):
    img_path_1 = './test_images/cautleya_spicata.jpg'
    img_path_2 = './test_images/hard-leaved_pocket_orchid.jpg'
    img_path_3 = './test_images/wild_pansy.jpg'
    img_path_4 = './test_images/orange_dahlia.jpg'
    files = img_path_1, img_path_2, img_path_3, img_path_4
    for image_path in files: 
        im = Image.open(image_path)
        test_image = np.asarray(im)
        processed_test_image = process_image(test_image)
        probs, classes = predict(image_path, model, 5)
        return probs, classes
    

if __name__ == "__main__":
    
    print ("start prediction ...")
    
    model = load_checkpoint(model_save)
    Sanity_Checkpoint(model)
    print ("end prediction..")