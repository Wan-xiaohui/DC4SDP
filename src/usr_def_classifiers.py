from collections import Counter
from sklearn.linear_model import LogisticRegression
import math
from scipy import *
from scipy.linalg import norm, pinv
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD


class Node:
    root = None

    def __init__(self, x, y, min_leaf=5, max_depth=5, depth=0):
        self.num_of_total_samples = len(x.index)
        self.col_names = list(x.columns)
        self.criterion = None
        self.left, self.right = None, None
        self.parent = None
        self.num_of_class_samples = Counter(y)
        self.x, self.y = x, y
        self.max_depth = max_depth
        self.depth = depth
        if len(np.unique(list(y))) > 1:
            self.lr = LogisticRegression().fit(x, list(y))
        self.min_leaf = min_leaf

    # a recursive method that builds the tree.
    # this method splits each node by the maximum gini impurity the node can get.
    def grow_tree(self):
        if self.parent == None:
            Node.root = self
        class_cnt = list(self.num_of_class_samples.values())
        if len(class_cnt) <= 1:
            return

        max_gini = self.gini_impurity(class_cnt[0], class_cnt[1])
        best_l, best_r = [], []
        for i in self.col_names:
            val, l, r = self.find_best_split(i)
            gini = self.gini_impurity(len(l), len(r))
            if gini > max_gini:
                max_gini = gini
                best_r, best_l = r, l
                self.criterion = (i, val)

        xl = pd.DataFrame(self.x, index=best_l, dtype="float")
        yl = pd.Series([int(self.y[j]) for j in best_l], index=[int(j) for j in best_l], dtype="float")
        xr = pd.DataFrame(self.x, index=best_r, dtype="float")
        yr = pd.Series([int(self.y[j]) for j in best_r], index=[int(j) for j in best_r], dtype="float")

        if len(xl) >= self.min_leaf and len(xr) >= self.min_leaf:
            self.depth += 1
            if self.depth >= self.max_depth:
                return
            else:
                self.left = Node(xl, yl, depth=self.depth, max_depth=self.max_depth, min_leaf=self.min_leaf)
                self.left.parent = self
                self.right = Node(xr, yr, depth=self.depth, max_depth=self.max_depth, min_leaf=self.min_leaf)
                self.right.parent = self
                self.left.grow_tree()
                self.right.grow_tree()
        else:
            return

    # a method that gets a name of a feature and returns the value of the feature that
    # gives the largest gini value and 2 lists of left child and right child indexes.
    def find_best_split(self, var_idx):
        col = self.x[var_idx]
        unique_vals = col.unique()
        max_gini = -1
        best_lhs, best_rhs = [], []
        for i in range(len(unique_vals)):
            lhs, rhs = [], []
            for j in zip(col.index, col):
                if j[1] <= unique_vals[i]:
                    lhs.append(j[0])
                else:
                    rhs.append(j[0])
            new_gini = self.get_gini_gain(lhs, rhs)

            if new_gini > max_gini:
                max_gini = new_gini
                best_lhs, best_rhs = lhs, rhs
                max_val = unique_vals[i]

        return max_val, best_lhs, best_rhs

    # a method that gets 2 lists of the indexes of the samples in each child node
    # and returns the gini gain of the parent node after the split
    def get_gini_gain(self, lhs, rhs):
        gini_before = self.gini_impurity(len(lhs), len(rhs))
        p_left = len(lhs) / (len(lhs) + len(rhs))
        p_right = len(rhs) / (len(lhs) + len(rhs))

        left_cnt = list(Counter([self.y[i] for i in lhs]).values())
        right_cnt = list(Counter([self.y[i] for i in rhs]).values())

        if len(left_cnt) == 1 and len(right_cnt) == 1:
            return gini_before
        elif len(left_cnt) == 1 or len(left_cnt) == 0:
            return gini_before - (p_right * self.gini_impurity(right_cnt[0], right_cnt[1]))
        elif len(right_cnt) == 0 or len(right_cnt) == 1:
            return gini_before - (p_left * self.gini_impurity(left_cnt[0], left_cnt[1]))
        else:
            return gini_before - (p_left * self.gini_impurity(left_cnt[0], left_cnt[1]) + p_right * self.gini_impurity(
                right_cnt[0], right_cnt[1]))

    # a method that checks if a node is a leaf
    def is_leaf(self):
        return True if self.left == None and self.right == None else False

    # a method that gets a dataframe that contains all the samples in a node
    # and returns a list of the predictions of each row in the dataframe using the predict_row method.
    def predict(self, x):
        return [self.predict_row(row) for index, row in x.iterrows()]

    # a method that gets a sample and returns its prediction by traversing the tree.
    # if the leaf node has only samples from one class return this class
    # else we use the logistic regression model we defined in init to predict the sample class.
    def predict_row(self, xi):
        node = Node.root
        while node != None:
            if node.is_leaf():
                num_of_classes_samples = Counter(node.y)
                if len(num_of_classes_samples.keys()) == 1:
                    return list(num_of_classes_samples.keys())[0]
                return node.lr.predict([xi])[0]
            else:
                if xi[node.criterion[0]] <= node.criterion[1]:
                    node = node.left
                else:
                    node = node.right

    # a static method that calculates the gini impurity of a node given the number of samples from each class
    @staticmethod
    def gini_impurity(y1_count, y2_count):
        return 1 - (math.pow(y1_count / (y1_count + y2_count), 2) + math.pow(y2_count / (y1_count + y2_count), 2))


class LogisticModelTree(object):
    def __init__(self, min_leaf=5, max_depth=5):
        self.min_leaf = min_leaf
        self.max_depth = max_depth

    def fit(self, X, y):
        cols = ['f' + str(i+1) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=cols)
        # y = pd.DataFrame(y)

        self.dtree = Node(X, y, self.min_leaf, self.max_depth)
        self.dtree.grow_tree()
        return self

    def predict(self, X):
        cols = ['f' + str(i+1) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=cols)
        y_pred = self.dtree.predict(X)
        y_pred = np.array(y_pred)
        return y_pred


# class RBFClassifier(BaseEstimator):
#     def __init__(self, k=2, n_neighbors=2, n_selection=2):
#         self.k = k
#         self.n_neighbors = n_neighbors
#         self.n_selection = n_selection
#
#     def euclidean_distance(self, x1, x2):
#         return np.linalg.norm(x1 - x2)
#
#     def rbf_hidden_layer(self, X):
#         def activation(x, c, s):
#             return np.exp(-self.euclidean_distance(x, c) / 2 * (s ** 2))
#
#         return np.array([[activation(x, c, s) for (c, s) in zip(self.cluster_, self.std_list_)] for x in X])
#
#     def fit(self, X, y):
#         def convert_to_one_hot(y, n_classes):
#             arr = np.zeros((y.size, n_classes))
#             arr[np.arange(y.size), y.astype(np.uint)] = 1
#             return arr
#
#         kmeans = KMeans(n_clusters=self.k, random_state=0)
#         self.cluster_ = kmeans.cluster_centers_
#         cond = self.k if self.n_neighbors > self.k or self.n_neighbors == 0 else self.n_neighbors
#
#         # Select N clusters centroids at "random"
#         if self.n_selection == 0:
#             self.std_list_ = np.array(
#                 [[self.euclidean_distance(c1, c2) for c1 in self.cluster_] for c2 in self.cluster_[: cond]])
#         else:
#             self.std_list_ = np.sort(
#                 np.array([[self.euclidean_distance(c1, c2) for c1 in self.cluster_] for c2 in self.cluster_]))
#
#             # Select N clusters centroids by distance (closest last)
#             if self.n_selection == 2:
#                 self.std_list_ = self.std_list_[::-1]
#
#             self.std_list_ = self.std_list_[:, : cond]
#         self.std_list_ = np.mean(self.std_list_, axis=1)
#         RBF_X = self.rbf_hidden_layer(X)
#         self.w_ = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ convert_to_one_hot(y, np.unique(y).size)
#
#     def predict(self, X):
#         rbs_prediction = np.array([np.argmax(x) for x in self.rbf_hidden_layer(X) @ self.w_])
#         return rbs_prediction
#
#     def get_params(self, deep=True):
#         return {"k": self.k, "n_neighbors": self.n_neighbors, "plot": self.plot, "n_selection": self.n_selection}
#
#
# class RBFNetClassifier:
#     """
#     https://github.com/kllaas/RBFNetwork/blob/master/rbf.py
#     """
#     def __init__(self, n_clusters, epochs=100):
#         self.n_clusters = n_clusters
#         self.w = []
#         self.error = []
#         self.h = 0
#         self.centers = []
#
#     def gaussian(self, x, media):
#         d_aux = 0
#         for d in range(len(x)):
#             d_aux += pow(x[d] - media[d], 2)
#         distancia = np.sqrt(d_aux)
#         return np.exp(distancia / (self.h ** 2))
#
#     def H(self, centers):
#         d_media = 0
#         for i in range(len(centers)):
#             for j in range(len(centers) - 1):
#                 d_aux = 0
#                 for d in range(len(centers[0])):
#                     d_aux += pow(centers[i][d] - centers[j][d], 2)
#                 d_media += np.sqrt(d_aux)
#         d_media = d_media / len(centers) * (len(centers) - 1)
#         return d_media / 2
#
#     def escolha_dos_centros(self, n_clusters, x_data):
#         centers = []
#         # print(n_clusters)
#         p = [0] * n_clusters
#         # seleciona os clusters aleatoriamente
#         for i in range(n_clusters):
#             p[i] = np.random.randint(0, len(x_data) - 1, 1)[0]
#             centers.append(x_data[p[i]])
#         self.centers = centers
#         return centers
#
#     def fit(self, x_data, y_data):
#         x_data, y_data = np.array(x_data), np.array(y_data)
#         testMatrixSingular = True
#         while testMatrixSingular:
#             # Parte I
#             centers = self.escolha_dos_centros(self.n_clusters, x_data)
#             # print(centers)
#             # parte II
#             # calcular paramentro h
#             self.h = self.H(centers)
#             # parte III
#             x_new = []
#             for i in range(len(x_data)):
#                 x_aux = []
#                 for j in range(len(centers)):
#                     # print(x_data[i],centers[j])
#                     x_aux.append(self.gaussian(x_data[i], centers[j]))
#                 x_new.append(x_aux)
#             # calculo dos pesos
#             x_new = np.array(x_new)
#
#             try:
#                 w = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_data)
#                 self.w = w
#                 testMatrixSingular = False
#             except np.linalg.LinAlgError:
#                 testMatrixSingular = True
#                 # print('matriz singular')
#
#     def predict(self, x):
#         y_pred = []
#         for i in range(len(x)):
#             x_new = []
#             for j in range(self.n_clusters):
#                 x_new.append(self.gaussian(x[i], self.centers[j]))
#             x_new = np.array(x_new)
#             y = x_new.dot(self.w)
#             if y < 0: y = 0
#             y_pred.append(int(round(y)))
#         return


class RBFNetClassifier(object):
    def __init__(self, hidden_layer_neurons=5):
        # self.centers = np.zeros((hidden_layer_neurons, input_dims))
        self.W = np.random.random((hidden_layer_neurons))
        self.hidden_layer_neurons = hidden_layer_neurons
        self.spreads = np.zeros((1, hidden_layer_neurons))
        self.model = Sequential()

    def gaussian_function(self, x, c, s):
        return np.exp(-1 * ((np.linalg.norm(x - c) / s) ** 2))

    def calc_activation(self, X):
        g_values = np.zeros((self.hidden_layer_neurons, X.shape[1]))
        for i in range(X.shape[1]):
            x = X[:, i]
            g = np.zeros((self.hidden_layer_neurons,))
            for c in range(self.hidden_layer_neurons):
                center = self.centers[c]
                s = self.spreads[c]
                g[c] = self.gaussian_function(x, center, s)
            g_values[:, i] = g
        return g_values

    def fit(self, X, Y):
        self.centers = KMeans(n_clusters=self.hidden_layer_neurons).fit(X.T).cluster_centers_
        avg_distance = 0
        for c in range(len(self.centers)):
            distance_to_nearest = max([np.linalg.norm(self.centers[c] - X[:, i]) for i in range(X.shape[1])])
            avg_distance += distance_to_nearest

        avg_distance /= self.hidden_layer_neurons
        self.spreads = np.repeat(avg_distance, self.hidden_layer_neurons)
        g_values = self.calc_activation(X)
        self.model.add(layers.Dense(1, input_dim=self.hidden_layer_neurons, activation='sigmoid'))
        sgd_optimizer = SGD()
        self.model.compile(loss='binary_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])
        self.model.fit(x=g_values.T, y=Y, epochs=500, shuffle=True)

    def predict(self, X):
        g_values = self.calc_activation(X)
        p = self.model.predict(g_values.T)
        return np.where(p > 0.5, 1, 0)

