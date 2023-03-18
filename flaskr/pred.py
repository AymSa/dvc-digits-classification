import sys
from flask import Blueprint, render_template, request
from src import module_from_file
import joblib

sys.path.append(".")
sys.path.append("..")

PREPROCESS_PATH = "src/preprocess.py"
PICKLE_PATH = 'models/model.joblib'

preprocess = module_from_file("preprocess", PREPROCESS_PATH)

bp = Blueprint("predict", __name__)


@bp.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        x = preprocess.preprocess_data_user(
            request.get_data()
        )
        
        pipe = joblib.load(PICKLE_PATH)
        pred = pipe.predict(x)

        return str(int(pred))

    return render_template("predict/hand_write.html")
