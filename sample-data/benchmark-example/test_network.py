from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier

dataset_df = pd.read_csv("./dataset.csv", sep="$")
dataset_df = pd.concat([dataset_df[dataset_df['class_value'] == 1],
                        dataset_df[dataset_df['class_value'] == 0].sample(128)], axis=0)
dataset_df.head(5)


def extract_features_keywords(X, keywords):
    result = []
    for x in X:
        result.append(np.array([1 if keyword in x else 0 for keyword in keywords]))

    return np.array(result).reshape(-1, len(keywords))


def evaluate_model(X_train, y_train, X_test):
    keywords = ['"', "sql", "statement", "select", "insert", "delete", "update", "drop", "execute"]

    # perform feature extraction on X_train
    X_train = extract_features_keywords(X_train, keywords)

    # train a model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # perform feature extraction on X_test
    X_test = extract_features_keywords(X_test, keywords)

    # infer
    return model.predict(X_test)


seeds = [2324, 343, 1232, 4343, 434343, 4343, 135, 343, 16, 999343]


def evaluate_loop(X, y, evaluate_model_func, runs=10):
    counter_y = Counter(y)
    results = dict(acc=[], rec=[], prec=[], fscore=[], cnf=[],
                   n=X.shape[0],
                   n_pos=counter_y[1], n_neg=counter_y[0])

    for run in range(runs):
        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seeds[run])
        kfold = KFold(n_splits=10, shuffle=True, random_state=seeds[run])
        print(f'Run {run + 1} / {runs}')
        cv_preds = []
        cv_trues = []
        print('- Fold: ', end=" ")
        fold = 1
        for train_index, val_index in skfold.split(X, y):
            print(f'{fold}', end=' ')
            y_pred = evaluate_model_func(X.iloc[train_index], y.iloc[train_index], X.iloc[val_index])

            cv_preds += y_pred.tolist()
            cv_trues += y.iloc[val_index].tolist()
            fold += 1
        print("")
        results['acc'].append(accuracy_score(cv_trues, cv_preds))
        results['rec'].append(recall_score(cv_trues, cv_preds))
        results['prec'].append(precision_score(cv_trues, cv_preds))
        results['fscore'].append(f1_score(cv_trues, cv_preds))
        results['cnf'].append(confusion_matrix(cv_trues, cv_preds))

    return results


res = evaluate_loop(X=dataset_df['contents'].astype(str),
                    y=dataset_df['class_value'],
                    evaluate_model_func=evaluate_model,
                    runs=10)
