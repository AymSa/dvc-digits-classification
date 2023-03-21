import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
import os
from prefect import flow 
import importlib.util

def module_from_file(module_name, file_path): ## BAD DESIGN 
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

METRICS_PATH = 'src/metrics.py'
metrics = module_from_file("metrics", METRICS_PATH)

def get_variables(data):  ## A mettre dans le fichier de preprocess
    return data.loc[:, 1:].values, data.loc[:, 0].values

@flow 
def train(data, isDataFrame=False):  # Model RFC
    if not isDataFrame:
        data = pd.read_csv(data, header=None)

    X_train, y_train = get_variables(data)

    ### Meilleur design de la pipeline à faire (features store)
    pipe = Pipeline(
        [
            ("scalar", StandardScaler()),
            (
                "RFC",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    criterion="gini",
                ),
            ),
        ]
    )

    logs = pipe.fit(X_train, y_train)

    return pipe, {"training_logs": logs}

def evaluate(data, pipe, OUTPUT_PATH, isDataFrame=False):
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if not isDataFrame:
        data = pd.read_csv(data, header=None)

    X_test, y_test = get_variables(data)

    y_pred_test = pipe.predict(X_test)

    labels = pipe.classes_
    test_results = metrics.eval_metrics(y_test, y_pred_test)

    y_scores = pipe.predict_proba(X_test)[:, 1]

    # ROC curve

    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, [0 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, y_scores)

    # precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    logs = {
        "metrics": test_results,
        "roc_curve": {
            "model_tpr": model_tpr,
            "model_fpr": model_fpr,
            "dummy_tpr": dummy_tpr,
            "dummy_fpr": dummy_fpr,
        },
        "precision_recall_curve": {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds,
        },
    }

    metrics.plot_roc(logs, OUTPUT_PATH, "RFC")  ## Brut code, mauvais design !
    metrics.plot_prc(logs, OUTPUT_PATH)
    metrics.plot_cf(y_test, y_pred_test, labels, OUTPUT_PATH)

    return logs
