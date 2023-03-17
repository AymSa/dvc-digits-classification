### CLI Arguments : Do not use argparse, incompatibility with prefect
import os
import sys
import joblib
from utils import *

DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]
PARAM = int(sys.argv[3])

sys.path.insert(1, MODEL_PATH)

model = module_from_file("model", MODEL_PATH)

if __name__ == "__main__":
    pipe, logs_train = model.train(DATA_PATH, PARAM)
    # Save pipeline

    joblib.dump(pipe, "./models/model.joblib")  ##PICKLE_PATH
