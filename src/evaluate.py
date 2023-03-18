### CLI Arguments : Do not use argparse, incompatibility with prefect
import os
import sys
import joblib
import json


def run_evaluation(data, model_ckpt, model):
    pipe = joblib.load(model_ckpt)
    logs_eval = model.evaluate(data, pipe, "./results")

    with open("./results/metrics.json", "w") as f_out:
        json.dump(logs_eval["metrics"], f_out)


if __name__ == "__main__":
    from __init__ import module_from_file 
    DATA_PATH = os.path.abspath(sys.argv[1])
    MODEL_PATH = sys.argv[2]
    PICKLE_PATH = sys.argv[3]

    sys.path.insert(1, MODEL_PATH)

    model = module_from_file("model", MODEL_PATH) ##From init

    run_evaluation(DATA_PATH, PICKLE_PATH, model)
