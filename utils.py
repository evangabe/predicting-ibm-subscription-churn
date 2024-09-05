import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import Booster
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score
import numpy as np

""" 
INFERENCE
"""

# Retrieve model from file
def get_model(model_name: str):
    if model_name == 'xgboost':
        model = XGBClassifier()
        model.load_model("./models/xgboost.json")
    elif model_name == 'catboost':
        model = CatBoostClassifier()
        model.load_model("./models/catboost")
    else:
        model = Booster(model_file="./models/lightgbm.txt")
    return model

# Run churn prediction on input
def predict_churn(model, x):
    try:
        return model.predict_proba(x)[:, 1]
    except AttributeError:
        return model.predict(x)

""" 
PLOTTING 
"""

# Plot Retriever Operating Characteristic Area-Under Curve (ROC-AUC)
def plot_roc_auc(model, X, y):
    preds = predict_churn(model, X)
    fpr, tpr, _ = roc_curve(y, preds)
    
    plt.plot(fpr, tpr, 'b', label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC Plot')
    plt.legend()

    return plt, roc_auc_score(y, preds)

"""
DATA
"""

def bin_value(feature_max, feature_min, value):
    bins = np.linspace(feature_min, feature_max, 4)
    if value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'