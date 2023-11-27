import tflite_runtime.interpreter as tflite
# from keras_image_helper import create_preprocessor
from PIL import Image
import requests
import numpy as np 

def get_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((299, 299), Image.Resampling.NEAREST)
    x = np.array(img, dtype='float32')
    return np.array([x])

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

# preprocessor = create_preprocessor('xception', target_size=(299, 299))

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
# memory allocation
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

# url = 'http://bit.ly/mlbookcamp-pants'

def predict(url):
    # X = preprocessor.from_url(url)
    img = get_image(url)
    X = preprocess_input(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    predictions = preds[0].tolist()

    return dict(zip(classes, predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
