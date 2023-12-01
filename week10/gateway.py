import os
import grpc
#import tensorflow as tf
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# flask
from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

# Tensorflow Serving connects on port 8500
# host = 'localhost:8500'

# if not default tf serving host, connect on 8500
# env var for docker compose
host = os.getenv("TF_SERVING_HOST", "localhost:8500")

# create a grpc channel. Insecure, because everything will be inside Kubernetes
channel = grpc.insecure_channel(host)
# initiate Prediction Service Stub
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

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


# translate NumPy array to TF serving protobuf
# replaced with tensorflow protobuf function
# def np_to_protobuf(data):
#     return tf.make_tensor_proto(data, shape=data.shape)

def prepare_X(url):
    ''' 
    Preprocess an image url into NumPy array ready for the model
    '''
    preprocessor = create_preprocessor('xception', target_size=(299, 299))
    X = preprocessor.from_url(url)

    return X

def prepare_request(X):
    # initialize a request
    pb_request = predict_pb2.PredictRequest()

    # specify model's name
    pb_request.model_spec.name = 'clothing-model'
    # specify signature name
    pb_request.model_spec.signature_name = 'serving_default'
    # specify input
    pb_request.inputs['input_2'].CopyFrom(np_to_protobuf(X))

    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_1'].float_val

    return dict(zip(classes, preds))

def predict(url):
    
    # Preprocess an image 
    X = prepare_X(url)
    pb_request = prepare_request(X)
    # get response from Docker stub contains host/port/channel
    pb_response = stub.Predict(pb_request, timeout=20.0)
    # output
    result = prepare_response(pb_response)

    return result

app = Flask('gateway')
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data["url"]
    result = predict(url)
    return jsonify(result)


if __name__ == "__main__":
    # docker test
    # url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(url)
    # print(response)

    # flask and docker-compose test
    app.run(debug=True, host='0.0.0.0', port=9696)