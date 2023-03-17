# Data

Here goes the data for the project, such as training datasets. When using DVC, it should point at this folder. 

mnist/mnist_train.csv : contains training images and labels from 0 to 9
mnist/mnist_test.csv : contains testing images and labels from 0 to 9

Created using src/preprocess_data.py
mnist/mnist_train_processed.csv : contains training images and labels (1 for number 0 and -1 for others)
mnist/mnist_test_processed.csv : contains testing images and labels (1 for number 0 and -1 for others)