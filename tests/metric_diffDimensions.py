from Generacion_datos import generate_data
import AdaptoLogit as al

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,  train_test_split
from sklearn import metrics

# Use Cross validation
X, y, beta = generate_data(n_obs=600, n_samples=100, n_important=50, scale=1, seed=44)
y = y.flatten()

# DF for metrics
list_weights = ["pca_1", "pca_pct", "spca", "svc", "xgb"]
df = pd.DataFrame(columns=["Euclidean distance betas", "TPR", "TNR", "CSR", "Elapsed time"], index=list_weights)

nsplits = 6
results = {}

for item in list_weights:
    print(item)
    accuracies = []
    tpr = []
    tnr = []
    e_dist = []
    elapsed_times = []

    for i in range(nsplits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        time_start = time.time()
        weight = al.AdaptiveWeights(power_weight=1, weight_technique=item)
        weight.fit(X_train, y_train)

        model = al.AdaptiveLogistic()
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'weight_array': weight.lasso_weights_,
                      'solver': ['liblinear', 'saga'], 'max_iter': [1000, 10000]}

        grid_search = GridSearchCV(model,
                                   param_grid,
                                   cv=3,
                                   scoring='accuracy',
                                   n_jobs=11)

        grid_search.fit(X_train, y_train)

        time_elapsed = (time.time() - time_start)
        elapsed_times.append(time_elapsed)

        final_model = grid_search.best_estimator_
        y_pred = final_model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        cm = metrics.confusion_matrix(y_test, y_pred)
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        tpr.append(TPR)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        tnr.append(TNR)

        euclidean_distance = np.linalg.norm(beta.T - final_model.coef_)
        e_dist.append(euclidean_distance)

    results["{}_{}".format(item, "accuracy")] = accuracies
    results["{}_{}".format(item, "euclidean_dist")] = e_dist

    df.loc[weight.weight_technique]["Euclidean distance betas"] = np.mean(e_dist)
    df.loc[weight.weight_technique]["TPR"] = np.mean(tpr)
    df.loc[weight.weight_technique]["TNR"] = np.mean(tnr)
    df.loc[weight.weight_technique]["CSR"] = np.mean(accuracies)
    df.loc[weight.weight_technique]["Elapsed time"] = np.mean(elapsed_times)

df