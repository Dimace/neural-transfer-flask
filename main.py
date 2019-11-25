#import tensorflow_core as tf
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, make_response, jsonify
from werkzeug.utils import secure_filename
import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import random

app = Flask(__name__, instance_relative_config=True)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'img')
os.environ['TFHUB_CACHE_DIR'] = os.path.join(APP_ROOT, 'tf_cache')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def stylizeImage(path_original_image, path_style_image, path_to_save):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    original_image = load_img(path_original_image)
    style_image = load_img(path_style_image)
    stylized_image = hub_module(tf.constant(original_image), tf.constant(style_image))[0]
    im = tensor_to_image(stylized_image)
    print("All done! Saving result as stylized.jpeg")
    print('path to save ' + path_to_save)
    index = random.randint(1, 10000)
    filename = "stylized" + str(index) + ".jpeg"
    im.save(path_to_save + '/' + filename, "JPEG")
    return filename

# a simple page that says hello
@app.route('/hello')
def hello():
    return 'Hello, World!'

@app.route('/img/<path:image_name>')
def img(image_name):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), image_name)

@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    if request.method == "POST":
        req = request
        originalImage = request.files["originalImage"]
        originalImagePath = os.path.join(app.config['UPLOAD_FOLDER'], originalImage.filename)
        originalImage.save(originalImagePath)
        styleImage = request.files["styleImage"]
        styleImagePath = os.path.join(app.config['UPLOAD_FOLDER'], styleImage.filename)
        styleImage.save(styleImagePath)
        print("Image saved")
        imageName = stylizeImage(originalImagePath, styleImagePath, app.config['UPLOAD_FOLDER'])
        res = make_response(jsonify({"imageName": "img/" + imageName}), 200)
        return res    

app.run(host='0.0.0.0')