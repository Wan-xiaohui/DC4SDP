import sys
sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef
from usr_def_classifiers import LogisticModelTree, RBFNetClassifier
from imodels.rule_set import SLIPPERClassifier
import wittgenstein as lw


# Algo Type 01: statistical techniques - NB
def NB(train_data, train_label):
    model = GaussianNB()
    model.fit(train_data, train_label)
    return model


# Algo Type 01:  Statistical techniques - Ridge
def Ridge(train_data, train_label):
    model = RidgeClassifier()
    model.fit(train_data, train_label)
    return model


# Algo Type 01:  Statistical techniques - LR
def LR(train_data, train_label):
    model = LogisticRegression()
    model.fit(train_data, train_label)
    return model


# Algo Type 02: Lazy-learning techniques - KNN
def KNN(train_data, train_label):
    model = KNeighborsClassifier()
    model.fit(train_data, train_label)
    return model


# Algo Type 03: Decision tree techniques - DT
def DT(train_data, train_label):
    model = DecisionTreeClassifier()
    model.fit(train_data, train_label)
    return model


# Algo Type 03: Decision tree techniques - LMT
def LMT(train_data, train_label):
    model = LogisticModelTree()
    model.fit(train_data, train_label)
    return model


# Algo Type 04: Rule-based Methods - RIPPER
def RIPPER(train_data, train_label):
    model = lw.RIPPER()
    model.fit(train_data, train_label)
    return model


# Algo Type 04: Rule-based Methods - IREP
def IREP(train_data, train_label):
    model = lw.IREP()
    model.fit(train_data, train_label)
    return model


# Algo Type 05: Support Vector Machine Methods - linearSVM
def linearSVM(train_data, train_label):
    model = SVC(kernel='linear')
    model.fit(train_data, train_label)
    return model


# Algo Type 05: Support Vector Machine Methods - rbfSVM
def kernelSVM(train_data, train_label):
    model = SVC()
    model.fit(train_data, train_label)
    return model


# Algo Type 06: Neural Network Methods - MLP
def MLPNet(train_data, train_label):
    model = MLPClassifier()
    model.fit(train_data, train_label)
    return model


# Algo Type 06: Neural Network Methods - RBFNet
def RBFNet(train_data, train_label):
    input_dims = train_data.shape[1]
    model = RBFNetClassifier(input_dims, hidden_layer_neurons=5)
    model.fit(train_data, train_label)
    return model

# Algo Type 07: Ensemble learning Methods - RF, GBDT, Bagging + CART/SVM/MLP/LR/NB, Boosting + CART/SVM/MLP/LR/NB
def Voting(train_data, train_label):
    model = VotingClassifier(estimators=[
        ("lr", LogisticRegression()),
        ("svm", SVC()),
        ("nb", GaussianNB()),
        ("mlp", MLPClassifier()),
        ("dt", DecisionTreeClassifier())])
    model.fit(train_data, train_label)
    return model


def RF(train_data, train_label):
    model = RandomForestClassifier()
    model.fit(train_data, train_label)
    return model


def GBDT(train_data, train_label):
    model = GradientBoostingClassifier()
    model.fit(train_data, train_label)
    return model


def BoostingDT(train_data, train_label):
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    model.fit(train_data, train_label)
    return model


def BoostingSVM(train_data, train_label):
    model = AdaBoostClassifier(base_estimator=SVC())
    model.fit(train_data, train_label)
    return model


def BoostingMLP(train_data, train_label):
    model = AdaBoostClassifier(base_estimator=MLPClassifier())
    model.fit(train_data, train_label)
    return model


def BoostingLR(train_data, train_label):
    model = AdaBoostClassifier(base_estimator=LogisticRegression())
    model.fit(train_data, train_label)
    return model


def BoostingNB(train_data, train_label):
    model = AdaBoostClassifier(base_estimator=GaussianNB())
    model.fit(train_data, train_label)
    return model


def BaggingDT(train_data, train_label):
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier())
    model.fit(train_data, train_label)
    return model


def BaggingSVM(train_data, train_label):
    model = BaggingClassifier(base_estimator=SVC())
    model.fit(train_data, train_label)
    return model


def BaggingLR(train_data, train_label):
    model = BaggingClassifier(base_estimator=LogisticRegression())
    model.fit(train_data, train_label)
    return model


def BaggingMLP(train_data, train_label):
    model = BaggingClassifier(base_estimator=MLPClassifier())
    model.fit(train_data, train_label)
    return model


def BaggingNB(train_data, train_label):
    model = BaggingClassifier(base_estimator=GaussianNB())
    model.fit(train_data, train_label)
    return model