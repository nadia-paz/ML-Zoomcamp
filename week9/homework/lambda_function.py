import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

# url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def get_X(url, target_size=(150, 150)):
    img = download_image(url)
    img = prepare_image(img, target_size)
    # rescale image
    rescaled = np.array(img, dtype="float32") / 255
    # prepare for the model and return
    return np.array([rescaled])

def predict(url):
    X = get_X(url)
    interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    # returns 2D array shape (1, 1)
    preds = interpreter.get_tensor(output_index)
    # return prediction's value
    # return preds[0].tolist()
    return float(preds[0][0])

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result