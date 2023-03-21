### CLI Arguments : Do not use argparse, incompatibility with prefect
import os
import sys
import joblib
from __init__ import module_from_file

DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]

sys.path.insert(1, MODEL_PATH)

model = module_from_file("model", MODEL_PATH)

def run_train(train_path):
    pipe, _ = model.train(train_path)
    # Save pipeline
    joblib.dump(pipe, "./models/model.joblib")  ##PICKLE_PATH

if __name__ == "__main__":
    run_train(DATA_PATH)