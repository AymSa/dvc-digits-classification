

```
dvc run -n preprocess \
  -d ./src/preprocess_data.py -d data/mnist/mnist_train.csv -d data/mnist/mnist_test.csv \
  -o data/mnist/mnist_train_processed.csv -o data/mnist/mnist_test_processed.csv \
  python ./src/preprocess_data.py data/mnist/mnist_train.csv data/mnist/mnist_test.csv
```

```
dvc run -n train \
  -d ./src/train.py -d ./data/mnist/mnist_train_processed.csv -d ./src/model.py \
  -o ./models/model.joblib \
  python ./src/train.py ./data/mnist/mnist_train_processed.csv ./src/model.py
```


```
dvc run -n evaluate -d ./src/evaluate.py -d ./data/mnist/mnist_test_processed.csv \
  -d ./src/model.py -d ./models/model.joblib -d ./src/metrics.py \
  -M ./results/metrics.json \
  -o ./results/precision_recall_curve.png -o ./results/roc_curve.png \
  python ./src/evaluate.py ./data/mnist/mnist_test_processed.csv ./src/model.py ./models/model.joblib
  ```