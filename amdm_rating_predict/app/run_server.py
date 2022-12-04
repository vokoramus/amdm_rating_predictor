import numpy as np  # dont delete! Needs for loaded model
import pandas as pd
import dill
from flask import Flask, request, jsonify
from time import strftime

import logging
from logging.handlers import RotatingFileHandler


app = Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


modelpath = "./models/rf_pipeline.dill"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=['POST'])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")

    # ensure the data was properly uploaded to our endpoint
    if request.method == "POST":
        request_json = request.get_json()  # получаем JSON, переданный через POST
        logger.info(f'{dt} Received: {len(request_json.items())} items')

        data_dict = request_json
        # data_dict = {k: [v] for (k, v) in request_json.items()}
        print(data_dict)
        df_ = pd.DataFrame.from_dict(data_dict, orient='index').T
        # df_ = pd.DataFrame().from_dict(data_dict)
        df_.iloc[0] = df_.iloc[0].apply(lambda x: eval(x))

        logger.info(f'{dt} Data: chords={data_dict["chords"]}')
        # prediction
        try:
            pred = model.predict(df_)[0]

        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return jsonify(data)

        data['predictions'] = pred
        logger.info(f'{dt} pred = {pred} \n')
        data["success"] = True
        print('OK')

    # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8180, debug=True)
