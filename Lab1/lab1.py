import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve
from pyod.models.knn import KNN

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

def ex1():
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Normal')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Outliers')
    plt.show()

def ex2():
    clf = KNN(contamination=0.3)
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
    plt.title('ROC Curve')
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


def mahalanobis(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

def ex4():
    X_train4, _, y_train4, _ = generate_data(n_train=1000, n_test=0, n_features=3, contamination=0.1)

    custom_mean = np.array([4, 3, 5])
    custom_variance_factor = 3 
    
    X_train4 = X_train4 * custom_variance_factor + custom_mean

    mean_vector = np.mean(X_train4, axis=0)
    cov_matrix = np.cov(X_train4, rowvar=False) + np.eye(X_train4.shape[1]) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mahal_distances = np.array([mahalanobis(x, mean_vector, inv_cov_matrix) for x in X_train4])

    threshold = np.quantile(mahal_distances, 0.9)

    y_pred = (mahal_distances > threshold).astype(int)

    balanced_acc = balanced_accuracy_score(y_train4, y_pred)
    print(f"Balanced Accuracy Ex4: {balanced_acc:.4f}\n")


if __name__=="__main__":
    ex1() 
    ex2()
    ex3()
    ex4()
