import numpy as np
from sklearn import neighbors

def distance_matrix(X1, X2):
    a1 = X1[:,None,:]
    a2 = X2[None,:,:]
    dist2 = (a1 - a2) ** 2
    dist2 = dist2.sum(axis=2)
    dist = np.sqrt(dist2)
    return dist

def kmin_argkmin(dist_matrix,K):
    kmin_ind = np.argsort(dist_matrix)[:, 0:K]
    kmin_val = np.sort(dist_matrix)[:, 0:K]
    return kmin_val, kmin_ind

def knn_classifier(X_train, label_train, X_test,K,weights="uniform",metric="euc"):
    dist_matrix = distance_matrix(X_test,X_train)
    knn_dist, knn_ind = kmin_argkmin(dist_matrix,K)
    knn_labels = label_train[knn_ind]

    label_max = label_train.max()
    votes = np.zeros((len(X_test),label_max+1))
    for i in range(len(X_test)):
        if weights == "distance":
            votes[i] = np.bincount(knn_labels[i], minlength=label_max+1, weights=1/knn_dist[i])
        else:
            votes[i] = np.bincount(knn_labels[i], minlength=label_max + 1)
    label_test = np.argmax(votes,axis=1)
    return label_test

def knn_classifier_sklearn(X_train, label_train, X_test,K,weights="uniform",metric="euc"):
    clf = neighbors.KNeighborsClassifier(K, weights=weights, algorithm="brute")
    clf.fit(X_train, label_train)

    label_test_sklearn = clf.predict(X_test)
    return label_test_sklearn

def calculate_accuracy(label_true, label_pred):
    acc = (label_pred == label_true).sum()
    return acc / len(label_true)