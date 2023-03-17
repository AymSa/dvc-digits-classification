import pandas as pd
import sys
import os

def preprocess_data(DATA_PATH):
    # Processing example : Multiclassification to binary classification 0 vs All
    df = pd.read_csv(DATA_PATH, header=None)

    df.loc[:, 0] = (
        2 * (df.loc[:, 0] == 0) - 1
    )

    processed_data_path = DATA_PATH.split(".")[0] + "_processed.csv"
    df.to_csv(processed_data_path, header=None, index=False)

    return processed_data_path


if __name__ == "__main__":
    DATA_PATH_TRAIN = os.path.abspath(sys.argv[1])
    DATA_PATH_TEST = os.path.abspath(sys.argv[2])
    
    processed_data_path = preprocess_data(DATA_PATH_TRAIN)
    print("Saved to {}".format(processed_data_path))

    processed_data_path = preprocess_data(DATA_PATH_TEST)
    print("Saved to {}".format(processed_data_path))
