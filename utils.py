import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
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
        model.load_model("models/xgboost.json")
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
def plot_roc_curve(model, X, y):
    """
    Given true labels `y`, plot the ROC Curve for model predictions on test set `X`.

    Args:
        model (Object): The model used to make predictions on test set `X`.
        X (Object): The test set DataFrame whose predictions will be compared to true labels `y`
        y (List[int]): The true labels (`0` = No Churn, `1` = Churn)
    
    Returns:
        (Object, float): Tuple containing Matplotlib figure and ROC-AUC score
    """

    preds = predict_churn(model, X)
    fpr, tpr, _ = roc_curve(y, preds)
    
    xy_range = [0, 1]
    plt.plot(fpr, tpr, 'b', label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot(xy_range, xy_range, 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(xy_range)
    plt.ylim(xy_range)
    plt.title('Retriever Operating Characteristic Curve')
    plt.legend()

    return plt, roc_auc_score(y, preds)

# Plot Precision-Recall Curve
def plot_precision_recall_curve(model, X, y):
    """
    Given true labels `y`, plot the PR Curve for model predictions on test set `X`.

    Args:
        model (Object): The model used to make predictions on test set `X`.
        X (Object): The test set DataFrame whose predictions will be compared to true labels `y`
        y (List[int]): The true labels (`0` = No Churn, `1` = Churn)
    
    Returns:
        (Object, float): Tuple containing Matplotlib figure and F1 score
    """

    preds = predict_churn(model, X)
    y_hat = np.mean(y)
    xy_range = [0, 1]
    plt.plot(xy_range, [y_hat, y_hat], 'r--', label='Trivial Model')
    
    p, r, _ = precision_recall_curve(y, np.mean(preds, axis=1).reshape(-1,1))
    plt.plot(r, p, 'b.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(xy_range)
    plt.ylim(xy_range)
    plt.title('Precision-Recall Curve')

    f1 = f1_score(y, preds)

    return plt, f1

"""
DATA
"""


# Binning for numerical features
def bin_value(feature_max, feature_min, value):
    bins = np.linspace(feature_min, feature_max, 4)
    if value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'

# Fit ordinal encoder to pretraining data
def get_encoder(pretrain, features):
    X_pretrain = pretrain.drop("Churn", axis=1)
    ordinal_encoder = OrdinalEncoder()
    X_pretrain[features] = ordinal_encoder.fit_transform(X_pretrain[features])
    return ordinal_encoder

# encode categorical features based on pretraining distributions
def encode(encoder, features, X):
    narr = [X[features].to_numpy()]
    print(narr)
    X[features] = encoder.transform(narr)[0]
    return X