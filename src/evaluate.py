### CLI Arguments : Do not use argparse, incompatibility with prefect 
import os 
import sys
import joblib
import json
from utils import *

DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]
PICKLE_PATH = sys.argv[3]

sys.path.insert(1, MODEL_PATH)

model = module_from_file("model", MODEL_PATH)

if __name__ == '__main__':

    pipe =  joblib.load(PICKLE_PATH)
    logs_eval = model.evaluate(DATA_PATH, pipe, "./results")

    with open("./results/metrics.json", "w") as f_out:
        json.dump(logs_eval["metrics"], f_out)
    