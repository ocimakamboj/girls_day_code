import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from knn_classifier import knn_classifier

def get_mesh_grid(X, h, margin=0.5):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx,yy

def plot_db_with_training(X_train,label_train,n_neighbors,weights='uniform',h=0.03,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)
    colors_db = np.array([[255, 204, 204], [153, 204, 255], [204, 255, 204]])
    cmap_db = ListedColormap(colors_db / 255)

    xx, yy = get_mesh_grid(X_train, h, margin=margin)
    X_db = np.c_[xx.ravel(), yy.ravel()]
    labels_db = knn_classifier(X_train, label_train, X_db, n_neighbors, weights)
    labels_db = labels_db.reshape(xx.shape)
    fig, ax = plt.subplots(tight_layout=True)
    ax.contourf(xx, yy, labels_db, cmap=cmap_db)
    ax.contour(xx, yy, labels_db, colors='k',linewidths=0.5)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=label_train, edgecolor='k', cmap=cmap_points, linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("Training Data, k = %i" % (n_neighbors))
    fig.show()

def plot_training(X_train,label_train,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)

    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=label_train, cmap=cmap_points, edgecolor='k',linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(X_train[:, 0].min()-margin, X_train[:, 0].max()+margin)
    ax.set_ylim(X_train[:, 1].min()-margin, X_train[:, 1].max()+margin)
    ax.set_title("Training data")
    fig.show()

def plot_test(X_test,label_test,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)

    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=label_test, cmap=cmap_points, edgecolor='k', linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(X_test[:, 0].min() - margin, X_test[:, 0].max() + margin)
    ax.set_ylim(X_test[:, 1].min() - margin, X_test[:, 1].max() + margin)
    ax.set_title("Test data")
    fig.show()

def plot_test_with_training(X_test,X_train,label_train,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)

    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=label_train, cmap=cmap_points, edgecolor='k', linewidths=0.2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c="yellow", edgecolor='k', linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(X_train[:, 0].min() - margin, X_train[:, 0].max() + margin)
    ax.set_ylim(X_train[:, 1].min() - margin, X_train[:, 1].max() + margin)
    ax.set_title("Training data, Test Data in Yellow")
    fig.show()

def plot_test_with_training_and_db(X_test, X_train,label_train,n_neighbors,weights='uniform',h=0.03,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)
    colors_db = np.array([[255, 204, 204], [153, 204, 255], [204, 255, 204]])
    cmap_db = ListedColormap(colors_db / 255)

    xx, yy = get_mesh_grid(X_train, h, margin=margin)
    X_db = np.c_[xx.ravel(), yy.ravel()]
    labels_db = knn_classifier(X_train, label_train, X_db, n_neighbors, weights)
    labels_db = labels_db.reshape(xx.shape)
    fig, ax = plt.subplots(tight_layout=True)
    ax.contourf(xx, yy, labels_db, cmap=cmap_db)
    ax.contour(xx, yy, labels_db, colors='k',linewidths=0.5)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=label_train, edgecolor='k', cmap=cmap_points, linewidths=0.2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c="yellow", edgecolor='k', linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("k = %i" % (n_neighbors))
    fig.show()

def plot_db_with_test(X_test, label_test, X_train,label_train,n_neighbors,weights='uniform',h=0.03,margin=0.5):
    colors_points = np.array([[204, 0, 0], [0, 102, 204], [0, 153, 0]])
    cmap_points = ListedColormap(colors_points / 255)
    colors_db = np.array([[255, 204, 204], [153, 204, 255], [204, 255, 204]])
    cmap_db = ListedColormap(colors_db / 255)

    xx, yy = get_mesh_grid(X_train, h, margin=margin)
    X_db = np.c_[xx.ravel(), yy.ravel()]
    labels_db = knn_classifier(X_train, label_train, X_db, n_neighbors, weights)
    labels_db = labels_db.reshape(xx.shape)
    fig, ax = plt.subplots(tight_layout=True)
    ax.contourf(xx, yy, labels_db, cmap=cmap_db)
    ax.contour(xx, yy, labels_db, colors='k',linewidths=0.5)
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=label_train, edgecolor='k', cmap=cmap_points, linewidths=0.2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=label_test, edgecolor='k', cmap=cmap_points, linewidths=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("Test Data, k = %i" % (n_neighbors))
    fig.show()

def plot_accuracy(acc_list, n, smooth_factor = 0.8):
    if smooth_factor == 0:
        plt.plot(n, acc_list)
        plt.xlabel("No. of neighbours")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        smoothed_acc = []
        alpha = smooth_factor
        smoothed_acc.append(acc_list[0])
        last = acc_list[0]
        for i in range(1, len(acc_list)):
            smoothed_val = (1 - alpha) * acc_list[i] + alpha * last
            last = smoothed_val
            smoothed_acc.append(smoothed_val)
        plt.plot(n, acc_list, c='C0')
        plt.plot(n, smoothed_acc, c='C0', alpha=0.4)
        plt.xlabel("No. of neighbours")
        plt.ylabel("Accuracy")
        plt.show()
