

```
dvc run -n preprocess \
  -d ./src/preprocess_data.py -d data/mnist/mnist_train.csv -d data/mnist/mnist_test.csv \
  -o data/mnist/mnist_train_processed.csv -o data/mnist/mnist_test_processed.csv \
  python ./src/preprocess_data.py data/mnist/mnist_train.csv data/mnist/mnist_test.csv
```