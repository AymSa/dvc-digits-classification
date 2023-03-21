import sys
from flask import Blueprint, render_template, request, jsonify
from src import module_from_file
import joblib
from pymongo import MongoClient
import os
import requests

sys.path.append(".")
sys.path.append("..")

PREPROCESS_PATH = "src/preprocess.py"
PICKLE_PATH = "models/model.joblib"
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

preprocess = module_from_file("preprocess", PREPROCESS_PATH)

bp = Blueprint("predict", __name__)


def save_to_db(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/digits", json=[rec])


@bp.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        record = {"imgb64": request.get_data()}

        x = preprocess.preprocess_data_user(record["imgb64"])

        pipe = joblib.load(PICKLE_PATH)
        proba = pipe.predict_proba(x)[0]
        prediction = int(pipe.predict(x))

        result = {"pred": int(prediction), "proba": round(proba[1], 3)}

        # save_to_db(record, prediction)
        # send_to_evidently_service(record, prediction)

        return jsonify(result)

    return render_template("predict/hand_write.html")
