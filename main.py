# This application is based on tensorflow guide https://www.tensorflow.org/tutorials/generative/style_transfer.
# Importing all needed modules for backend and image processing
# To run the application just navigate to the project's direcory and enter:
# python3 main.py
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, make_response, jsonify
from werkzeug.utils import secure_filename
import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import random

# creating Flask application
app = Flask(__name__, instance_relative_config=True)

# APP_ROOT is a variable that contains absolute path to the project's directory
# The absolute path is used here to prevent possible errors or mistakes caused by relative paths in functions.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# UPLOAD_FOLDER defines a directory for image upload. Images are also served from this directory.
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'img')

# os.environ['TFHUB_CACHE_DIR'] defines an environmental variable for tensorflow_hub module.
# By default tensorflow_hub uploads neural models to a system directory,
# so it may cause errors or freezes if your user doesn't have acces to them.
os.environ['TFHUB_CACHE_DIR'] = os.path.join(APP_ROOT, 'tf_cache')

# defines UPLOAD_FOLDER in the appliction's configuration map
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This function loads and prepares an image for style transfer.
# This function load an image and limit its maximum dimension to 512 pixels.
# It takes path to the image as an argument.
def load_img(path_to_img):

    max_dim = 512

    # reads file by path
    img = tf.io.read_file(path_to_img)

		# decodes an image into 3 channels. Returnds a 3-D tensor of type uint8
    img = tf.image.decode_image(img, channels=3)

		# converts the acquired tensor into a tensor of type float32 
    img = tf.image.convert_image_dtype(img, tf.float32)

		# shape variable gets size of 2-D image
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

		# long_dim variable gets the longest dimension of the 2-D shape
    long_dim = max(shape)

		# gets scale coefficient
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
		# adds a batch dimension to the image tensor. It is need only for convolutional networks
		# Basically batch dimension contains instances to work with.
		img = img[tf.newaxis, :]
    return img

# This function transfers style from one image to another and saves a result image to a specified directory.
# The function returns a name of the saved image.
# It takes paths of an original image and an image for stylization (path_original_image, path_style_image), 
# then it saves result by path_to_save
def stylizeImage(path_original_image, path_style_image, path_to_save):

		# hub_module gets a tensorfow model for style transfer, which is loaded from tensorflow hub
		hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

		# loading original and stylization images
    original_image = load_img(path_original_image)
    style_image = load_img(path_style_image)

		# stylized_image gets a resulting image from the loaded model
    stylized_image = hub_module(tf.constant(original_image), tf.constant(style_image))[0]
    im = tensor_to_image(stylized_image)

    print("All done! Saving result as stylized.jpeg")
    print('path to save directory' + path_to_save)

    index = random.randint(1, 10000)
		# In this application I didn't use any DB.
		# I needed to avoid getting cached images in my application when it gets images by the same name.
		# In addition, I wanted to see all my previous results.
		# So I decided to save images by a name "stylized + random number"
    filename = "stylized" + str(index) + ".jpeg"
    im.save(path_to_save + '/' + filename, "JPEG")
    return filename

# This function turns an input tensor of float type into an image.
# It takes a tensor as an argument.
def tensor_to_image(tensor):
	# multiplies elements of tensor by 255
  tensor = tensor*255

	# converts the tensor to uint8 type to make an image from it.
  tensor = np.array(tensor, dtype=np.uint8)

	# error handler
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


# This section contains all request handlers. Basically it is a part of web server functionality.

# This handler returns an image that is saved by path '/img/<path:image_name>'
@app.route('/img/<path:image_name>')
def img(image_name):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), image_name)

# This handler takes post requests by URL 'localhost:5000/style-transfer'
# Requests must contain 2 binary fields with names "originalImage" and "styleImage" (obviously)
# It responses with a json that contains "imageName" field.
# By this name frontend application can get a result image.
@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    if request.method == "POST":

				# gets originalImage data from the request and defines a path to upload this image.
        originalImage = request.files["originalImage"]

				#absolute path for uploading the original image
        originalImagePath = os.path.join(app.config['UPLOAD_FOLDER'], originalImage.filename)
				# saves the image by an absolute path.
        originalImage.save(originalImagePath)

				# this part does the same steps but for "styleImage"
        styleImage = request.files["styleImage"]
        styleImagePath = os.path.join(app.config['UPLOAD_FOLDER'], styleImage.filename)
        styleImage.save(styleImagePath)
        print("Image saved")

				# gets a name of a stylized image and returns it in a response.
        imageName = stylizeImage(originalImagePath, styleImagePath, app.config['UPLOAD_FOLDER'])
        res = make_response(jsonify({"imageName": "img/" + imageName}), 200)
        return res    

# it runs the application on port 5000 by default and makes it listen to all income connections
app.run(host='0.0.0.0')