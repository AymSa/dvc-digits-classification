import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_metrics(y, y_pred):
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return {"accuracy": acc, "recall": recall, "precision": precision, "f1": f1}

def plot_roc(logs, OUTPUT_PATH, model_name):
    plt.plot(
        logs["roc_curve"]["dummy_fpr"],
        logs["roc_curve"]["dummy_tpr"],
        linestyle="--",
        label="Dummy Classifer",
    )
    plt.plot(
        logs["roc_curve"]["model_fpr"],
        logs["roc_curve"]["model_tpr"],
        marker=".",
        label=model_name,
    )
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    out_path = OUTPUT_PATH + "/roc_curve.png"
    plt.savefig(out_path, dpi=80)
    plt.cla()


def plot_prc(logs, OUTPUT_PATH):
    precisions = logs["precision_recall_curve"]["precisions"]
    recalls = logs["precision_recall_curve"]["recalls"]
    thresholds = logs["precision_recall_curve"]["thresholds"]

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    out_path = OUTPUT_PATH + "/precision_recall_curve.png"
    plt.savefig(out_path, dpi=80)
    plt.cla()
