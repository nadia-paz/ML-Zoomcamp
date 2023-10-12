import pickle
from flask import Flask, request, jsonify

dv_file = 'dv.bin'
model_file = 'model1.bin'
app = Flask('credit')

with open(dv_file, 'rb') as f_dv:
    dv = pickle.load(f_dv)
f_dv.close()
print('DictVectorizer created')

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)
f_model.close()
print('Model Loaded')

@app.route("/credit_predict", methods=["POST"])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_hat = model.predict_proba(X)[0, 1]
    approved = y_hat >= 0.5
    result = {
        "approved" : bool(approved),
        "probability" : round(float(y_hat), 3)
    }

    return jsonify(result)

if __name__ == "main":
    app.run(debug=True, host='0.0.0.0', port=2912)
