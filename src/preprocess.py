import pandas as pd
import sys
import os
import re
import base64
from PIL import Image
import io
import numpy as np 
import matplotlib.pyplot as plt 

def preprocess_data_csv(DATA_PATH):
    # Processing example : Multiclassification to binary classification 0 vs All
    df = pd.read_csv(DATA_PATH, header=None)

    df.loc[:, 0] = (
        2 * (df.loc[:, 0] == 0) - 1
    )

    processed_data_path = DATA_PATH.split(".")[0] + "_processed.csv"
    df.to_csv(processed_data_path, header=None, index=False)

    return processed_data_path


def preprocess_data_user(user_request):
    imgstr = re.search(b"base64,(.*)", user_request).group(1)
    base64_decoded = base64.decodebytes(imgstr)
    data = Image.open(io.BytesIO(base64_decoded))
    x = data.resize((28, 28))
    x = np.invert(x)
    x = x[:, :, 0]
    # x = x / 255.
    # x[x > 0.35] = 1.
    
    plt.imsave('instance/process_img.jpg', x, cmap = 'gray')

    return x.flatten().reshape(1,-1)


if __name__ == "__main__":
    DATA_PATH_TRAIN = os.path.abspath(sys.argv[1])
    DATA_PATH_TEST = os.path.abspath(sys.argv[2])
    
    processed_data_path = preprocess_data_csv(DATA_PATH_TRAIN)
    print("Saved to {}".format(processed_data_path))

    processed_data_path = preprocess_data_csv(DATA_PATH_TEST)
    print("Saved to {}".format(processed_data_path))
