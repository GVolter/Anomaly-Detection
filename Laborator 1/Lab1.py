import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve
from pyod.models.knn import KNN

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

def ex1():
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Normal')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Outliers')
    plt.show()

def ex2(contamination_level):
    clf = KNN(contamination=contamination_level)
    clf.fit(X_train)
    ytrain = clf.predict(X_train)
    ytest = clf.predict(X_test)

    ytrainscore = clf.predict_proba(X_train)[:, 1]
    ytestscore= clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_train, ytrain).ravel()
    tnt, fpt, fnt, tpt = confusion_matrix(y_test, ytest).ravel()

    print(f"\nTraining Confusion Matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Testing Confusion Matrix: TN: {tnt}, FP: {fpt}, FN: {fnt}, TP: {tpt}")

    balanced_acc_train = (tp/(tp+fn) + tn/(tn+fp))/2
    balanced_acc_test = (tpt/(tpt+fnt) + tnt/(tnt+fpt))/2

    print(f"Balanced Accuracy Ex2 (Train): {balanced_acc_train:.4f}")
    print(f"Balanced Accuracy Ex2 (Test): {balanced_acc_test:.4f}\n")

    fpr_train, tpr_train, _ = roc_curve(y_train, ytrainscore)
    fpr_test, tpr_test, _ = roc_curve(y_test, ytestscore)   

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label="Train ROC")
    plt.plot(fpr_test, tpr_test, 'r--', label="Test ROC")
    plt.title('ROC Curve ')
    plt.legend()
    plt.show()

def ex3():
    X_train3, _, y_train3, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)
    z_scores = (X_train3 - np.mean(X_train3)) / np.std(X_train3)
    z_scores = np.abs(z_scores)
    threshold = np.quantile(z_scores, 0.9)


    y_pred = (z_scores > threshold).astype(int)

    balanced_acc = balanced_accuracy_score(y_train3, y_pred)
    print(f"Balanced Accuracy Ex3: {balanced_acc:.4f}\n")

def ex4():
    data = np.random.default_rng(0)
    d = 3
    mu = np.array([1., 3., 2.])
    sigma = np.array([
        [3., 1.2, 0.8],
        [1.2, 2.5, 0.4],
        [0.8, 0.4, 1.8]
    ])
    L = np.linalg.cholesky(sigma)

    n_out = int(np.round(100))
    n_in = 1000 - n_out

    x_in = data.standard_normal((n_in, d))
    y_in = x_in @ L.T + mu

    mu_out = mu + np.array([6., -4., 5.])
    sigma = 6.0 * sigma
    L_out = np.linalg.cholesky(sigma)
    
    x_out = data.standard_normal((n_out, d))
    y_out = x_out @ L_out.T + mu_out

    X = np.vstack([y_in, y_out])
    y_true = np.hstack([np.zeros(n_in, dtype=int), np.ones(n_out, dtype=int)])

    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    Z = (X - mean) / std

    z_score = np.abs(Z).max(axis=1)

    # print(f"Z-scores : {z_score}")

    threshold = np.quantile(z_score, 0.9)
    y_pred = (z_score > threshold).astype(int)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy Ex4: {bal_acc:.4f}")

    return X, y_true, y_pred, threshold
   

if __name__=="__main__":
    ex1()
    ex2(contamination_level=0.1)
    ex2(contamination_level=0.4)
    ex3()
    ex4()