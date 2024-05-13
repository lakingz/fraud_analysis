import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
##################################################
def create_features(df, labels=None):

    # Ensure label is provided
    if labels is None:
        raise ValueError("Label parameter must be provided.")
        return df
    for label in labels:
        # Convert the date_series to datetime if not already
        date_series = pd.to_datetime(df[label], errors='coerce')
        # Extract features
        df['year '+label] = date_series.dt.year
        df['month '+label] = date_series.dt.month
        df['day '+label] = date_series.dt.day
        df['day_of_week '+label] = date_series.dt.dayofweek  # Monday=0, Sunday=6
        df['is_weekend '+label] = date_series.dt.dayofweek > 4  # Boolean: True if Saturday or Sunday
        df.drop([label], axis=1, inplace=True)
    return df
##############################################
############################################## in case someone have problem with graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Adjust path as necessary
import graphviz
dot = graphviz.Digraph(comment='The Round Table')
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')
print(dot.source)
# Assuming you have the correct path and all installations set
try:
    dot.render('doctest-output/round-table.gv').replace('\\', '/')
except Exception as e:
    print("An error occurred:", e)
############################################## train test split
credit_raw = pd.read_excel('Data_Analysis_and_Visualization_segment.xlsx')
credit_raw.shape
credit_raw.head()
credit_raw.describe()
credit_raw.info()
credit_raw.isnull().sum()
credit = credit_raw
time_veriable = list(credit.select_dtypes('datetime64[ns]'))
create_features(credit, labels=time_veriable)
credit.info()

X = credit.drop(['Current Status'], axis=1)
X = pd.get_dummies(X)
X.info()
y = credit['Current Status']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.4, random_state=21)
X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=21)

#
# perfrom xgboost
#
from sklearn.metrics import accuracy_score
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

params = {
 "model_params": {
    "gamma": Real(0, 0.5, 'uniform'),
    "learning_rate": Real(0.03, 0.3, 'uniform'),    # default 0.1
    "max_depth": Integer(2, 6),                     # default 3
    "n_estimators": Integer(100, 150),              # default 100
  },
  "train_params": {
    "eval_metric": [
      "logloss",
      "error",
      "auc",
    ]
  }
}
############################ parameter search
clf_xgb_search = xgb.XGBClassifier()
search = BayesSearchCV(clf_xgb_search, search_spaces=params['model_params'], random_state=42, n_iter=32, cv=3, verbose=1, n_jobs=1,
                            return_train_score=True)
search.fit(X_train, y_train)
search.best_estimator_
search.best_params_

############################ model fitting
clf_xgb = xgb.XGBClassifier(**search.best_params_, use_label_encoder=False)
clf_xgb.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          **params["train_params"])

y_pred_train = clf_xgb.predict(X_train)
y_pred_valid = clf_xgb.predict(X_valid)
y_pred_test = clf_xgb.predict(X_test)
#
#
#

from sklearn.metrics import confusion_matrix
key_metric = pd.DataFrame()
cms = [confusion_matrix(y_train, y_pred_train), confusion_matrix(y_valid, y_pred_valid), confusion_matrix(y_test, y_pred_test)]
for cm in cms:
    #[0,0] is true negative, [0,1] is false positive, [1,0] is false negative, [1,1] is true positive
    TNR = (cm[0, 0])/(cm[0, 1]+cm[0, 0])
    TPR = (cm[1, 1])/(cm[1, 1]+cm[1, 0])
    Precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    F1 = 2/(1/Precision+1/TPR)
    #Fbeta = (1+0.5**2)*Precision*TPR/(0.5**2*Precision+TPR)
    Accuracy = (cm[0, 0]+cm[1, 1])/(cm[0, 0]+cm[0, 1]+cm[1, 0]+cm[1, 1])
    key_metric = key_metric._append({'TNR': TNR,
                                     'TPR/Recall': TPR,
                                     'Precision': Precision,
                                     'F1': F1,
                                     'Accuracy': Accuracy}, ignore_index = True)
key_metric['Data set'] = ['training set', 'valid set', 'test set']
key_metric

#
# curve
#
from sklearn.metrics import PrecisionRecallDisplay
matplotlib.use('Agg')
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    display = PrecisionRecallDisplay.from_estimator(
        clf_xgb, X_train, y_train, name="train", ax=ax
    )
    PrecisionRecallDisplay.from_estimator(
        clf_xgb, X_valid, y_valid, name="valid", ax=ax
    )
    PrecisionRecallDisplay.from_estimator(
        clf_xgb, X_test, y_test, name="test", plot_chance_level=True, ax=ax
    )
    ax.set_title("Precision-Recall curve")
    ax.set_title("Precision-Recall curve")
    fig.savefig('Precision-Recall curve.png')  # Save the figure
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')
#
#calibration curve
#

from sklearn.calibration import calibration_curve
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    y_probs_train = clf_xgb.predict_proba(X_train)[:, 1]
    true_prob_train, predicted_prob_train = calibration_curve(y_train, y_probs_train, n_bins=10)
    ax.plot(predicted_prob_train, true_prob_train, marker='o', linewidth=1, label='train')

    y_probs_valid = clf_xgb.predict_proba(X_valid)[:, 1]
    true_prob_valid, predicted_prob_valid = calibration_curve(y_valid, y_probs_valid, n_bins=10)
    ax.plot(predicted_prob_valid, true_prob_valid, marker='o', linewidth=1, label='valid')

    y_probs_test = clf_xgb.predict_proba(X_test)[:, 1]
    true_prob_test, predicted_prob_test = calibration_curve(y_test, y_probs_test, n_bins=10)
    ax.plot(predicted_prob_test, true_prob_test, marker='o', linewidth=1, label='test')

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    fig.savefig('Calibration Curve.png')  # Save the figure
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

#
#feature importance plot
#

importances = clf_xgb.feature_importances_
feature_names = X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
# Sorting the features based on importance
indices = np.argsort(importances)[::-1]
try:
    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(range(len(importances[0:10])), importances[indices[0:10]], color='b', align='center')
    ax.set_xticks(range(len(importances[0:10])))
    ax.set_xticklabels(feature_names[indices[0:10]], rotation=45, ha='right', fontsize=10)
    ax.set_title('Top 10 Feature Importances')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('Top 10 Feature Importances.png')  # Save the figure
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

try:
    # Creating the plot
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.bar(range(len(importances[1:50])), importances[indices[1:50]], color='b', align='center')
    ax.set_xticks(range(len(importances[1:50])))
    ax.set_xticklabels(feature_names[indices[1:50]], rotation=45, ha='right', fontsize=8)
    ax.set_title('Top 30 Feature Importances without initial RIS decline')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('Top 50 Feature Importances without initial RIS decline.png')  # Save the figure
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

#
#shap
#
import shap
try:
    explainer = shap.Explainer(clf_xgb)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values.values, X_train, show=False, plot_size=(10, 8))
    plt.savefig('shap_summary_default.png', bbox_inches='tight')  # Save the figure shap_summary_default
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

#
#plot tree
#

try:
    xgb.plot_tree(clf_xgb, num_trees=1, rankdir='LR')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 20)
    fig.savefig('tree1.png')
    plt.clf()
    xgb.plot_tree(clf_xgb, num_trees=2, rankdir='LR')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 20)
    fig.savefig('tree2.png')
    plt.clf()
    xgb.plot_tree(clf_xgb, num_trees=3, rankdir='LR')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 20)
    fig.savefig('tree3.png')
    plt.clf()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

