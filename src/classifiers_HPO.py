import numpy as np
import sys
sys.dont_write_bytecode = True
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, space_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import pandas as pd
from imodels import GreedyRuleListClassifier, BoostedRulesClassifier
from functools import partial


N_SPLITS = 5
MAX_EVALS = 20


class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype="float")

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype="float")
        y_train_resampled = np.zeros((len(y_train)), dtype="int")
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and hasattr(self, "classes_")))


def calc_performance(label_true, label_pred):
    MCC = matthews_corrcoef(label_true, label_pred)
    return MCC


def KNN_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def KNN_hyperopt_train_val(params):
        clf = neighbors.KNeighborsClassifier(**params)
        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)
            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_neighbors': hp.randint('n_neighbors', 1, 11),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'p': hp.randint('p', 1, 6),
        'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree'])
    }

    if opt_algo == 'RAND':
        best = fmin(KNN_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(KNN_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    else:
        print('PLEASE SET YOUR OPTIMIZATION ALGORITHM !!!')

    params = space_eval(param_space, best)
    model_tune = neighbors.KNeighborsClassifier(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def NB_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def NB_hyperopt_train_val(params):
        clf = GaussianNB(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'var_smoothing': hp.loguniform('var_smoothing', -10, -1)
    }

    if opt_algo == 'RAND':
        best = fmin(NB_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(NB_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)
    model_tune = GaussianNB(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def CART_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def CART_hyperopt_train_val(params):
        clf = DecisionTreeClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                   'min_samples_split': hp.randint('min_samples_split', 2, 6),
                   'min_samples_leaf': hp.randint('min_samples_leaf', 1, 6),
                   }

    if opt_algo == 'RAND':
        best = fmin(CART_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(CART_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)
    model_tune = DecisionTreeClassifier(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def LR_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def LR_hyperopt_train_val(params):
        clf = LogisticRegression(**params)
        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'C': hp.loguniform('C', -10, 10),
        'penalty': 'elasticnet',
        'solver': 'saga',
        'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    }

    if opt_algo == 'RAND':
        best = fmin(LR_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(LR_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)
    model_tune = LogisticRegression(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune

    # pred_proba = model_tune.predict_proba(test_data)[:, 1]
    # pred_y = model_tune.predict(test_data)
    #
    # prec, recall, false_alarm, auc_value, F_measure, G_measure, bal = calc_performance(test_label, pred_y, pred_proba)
    #
    # if metrics == 'F_measure':
    #     result = F_measure
    #
    # elif metrics == 'AUC_PR':
    #     result = auc_value
    #
    # elif metrics == 'G_measure':
    #     result = G_measure
    #
    # elif metrics == 'Bal_value':
    #     result = bal
    #
    # else:
    #     print('PLEASE SELECT THE METRICS !!!')
    #
    # return model_tune, result


def SVM_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def SVM_hyperopt_train_val(params):
        clf = SVC(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
        'C': hp.lognormal('C', -1, 11),
        'max_iter': hp.choice('max_iter', [10 * i for i in range(5, 21)]),
        'gamma': hp.choice('gamma', [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 'auto']),
        'probability': True
    }

    if opt_algo == 'RAND':
        best = fmin(SVM_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(SVM_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)
    model_tune = SVC(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def MLP_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def MLP_hyperopt_train_val(params):
        clf = MLPClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'alpha': hp.loguniform('alpha', -10, -1),
        'max_iter': hp.choice('max_iter', [10*i for i in range(5, 21)]),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(i,) for i in range(5, 12, 2)] +
                                        [(i, j) for i in range(5, 12, 2) for j in range(5, 12, 2)])
    }

    if opt_algo == 'RAND':
        best = fmin(MLP_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(MLP_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)

    model_tune = MLPClassifier(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def Boosted_RS_Opt(train_data, train_label, cols, metrics='MCC', opt_algo='TPE'):

    def Boosted_RS_hyperopt_train_val(params):
        n_estimators = params['n_estimators']
        del params['n_estimators']

        clf = BoostedRulesClassifier(estimator=partial(DecisionTreeClassifier, **params),
                                     n_estimators=n_estimators
                                     )

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)
            train_X = pd.DataFrame(train_X, columns=cols)
            val_X = pd.DataFrame(val_X, columns=cols)
            train_y = pd.DataFrame(train_y).astype('int')
            val_y = pd.DataFrame(val_y).astype('int')

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)
            pred_y = np.array(pred_y, dtype=int)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
                   'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                   'max_depth': hp.choice('max_depth', [1, 2, 3])
                   }

    if opt_algo == 'RAND':
        best = fmin(Boosted_RS_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(Boosted_RS_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)

    n_estimators = params['n_estimators']
    del params['n_estimators']

    model_tune = BoostedRulesClassifier(estimator=partial(DecisionTreeClassifier, **params),
                                        n_estimators=n_estimators
                                        )

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    train_data = pd.DataFrame(train_data, columns=cols)
    train_label = pd.DataFrame(train_label).astype('int')
    model_tune.fit(train_data, train_label)

    return model_tune


def Greedy_RL_Opt(train_data, train_label, cols, metrics='MCC', opt_algo='TPE'):

    def Greedy_RL_hyperopt_train_val(params):
        clf = GreedyRuleListClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            train_X = pd.DataFrame(train_X, columns=cols)
            val_X = pd.DataFrame(val_X, columns=cols)

            train_y = pd.DataFrame(train_y).astype('int')
            val_y = pd.DataFrame(val_y).astype('int')

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_depth': hp.choice('max_depth', [1, 2, 3, 4, 5])
                   }

    if opt_algo == 'RAND':
        best = fmin(Greedy_RL_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(Greedy_RL_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)

    model_tune = GreedyRuleListClassifier(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)
    train_data = pd.DataFrame(train_data, columns=cols)
    train_label = pd.DataFrame(train_label).astype('int')

    model_tune.fit(train_data, train_label)

    return model_tune


def RF_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def RF_hyperopt_train_val(params):
        clf = RandomForestClassifier(**params)

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)
            pred_y = clf.predict(val_X)
            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.7]),
        'min_samples_split': hp.randint('min_samples_split', 2, 6),
        'warm_start': True
    }

    if opt_algo == 'RAND':
        best = fmin(RF_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(RF_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)
    model_tune = RandomForestClassifier(**params)

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

    model_tune.fit(train_data, train_label)

    return model_tune


def SVMBoosting_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def SVMBoosting_hyperopt_train_val(params):
        n_estimators = params['n_estimators']
        del params['n_estimators']

        clf = AdaBoostClassifier(SVC(**params),
                                 n_estimators=n_estimators
                                 )

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
        # 'learning_rate': hp.choice('learning_rate', [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]),
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
        'C': hp.lognormal('C', -1, 11),
        'max_iter': hp.choice('max_iter', [10 * i for i in range(5, 21)]),
        'gamma': hp.choice('gamma', [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 'auto']),
        'probability': True
    }

    if opt_algo == 'RAND':
        best = fmin(SVMBoosting_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(SVMBoosting_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)

    n_estimators = params['n_estimators']
    del params['n_estimators']

    model_tune = AdaBoostClassifier(SVC(**params),
                                    n_estimators=n_estimators
                                    )

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)
    model_tune.fit(train_data, train_label)

    return model_tune


def MLPBoosting_Opt(train_data, train_label, metrics='MCC', opt_algo='TPE'):
    def MLPBoosting_hyperopt_train_val(params):
        n_estimators = params['n_estimators']
        del params['n_estimators']
        # learning_rate = params['learning_rate']
        # del params['n_estimators'], params['learning_rate']

        clf = AdaBoostClassifier(customMLPClassifer(**params),
                                 n_estimators=n_estimators
                                 )

        k_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

        metrics_ = []
        for train_index, val_index in k_folds.split(train_data, train_label):
            train_X, val_X = train_data[train_index], train_data[val_index]
            train_y, val_y = train_label[train_index], train_label[val_index]

            train_X, train_y = SMOTE(random_state=0).fit_sample(train_X, train_y)

            clf.fit(train_X, train_y)

            # pred_proba = clf.predict_proba(val_X)[:, 1]
            pred_y = clf.predict(val_X)

            mcc = calc_performance(val_y, pred_y)

            if metrics == 'MCC':
                metrics_.append(mcc)

            else:
                print('PLEASE SELECT THE METRICS !!!')
                break

        return {
            'loss': 1 - np.mean(metrics_),
            'status': STATUS_OK
        }

    param_space = {
        'n_estimators': hp.choice('n_estimators', [10 * i for i in range(1, 6)]),
        'alpha': hp.loguniform('alpha', -10, -1),
        'max_iter': hp.choice('max_iter', [10 * i for i in range(5, 21)]),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(i,) for i in range(5, 12, 2)] +
                                        [(i, j) for i in range(5, 12, 2) for j in range(5, 12, 2)])
    }

    if opt_algo == 'RAND':
        best = fmin(MLPBoosting_hyperopt_train_val, param_space, algo=rand.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    elif opt_algo == 'TPE':
        best = fmin(MLPBoosting_hyperopt_train_val, param_space, algo=tpe.suggest, max_evals=MAX_EVALS, show_progressbar=False)

    params = space_eval(param_space, best)

    n_estimators = params['n_estimators']
    del params['n_estimators']

    model_tune = AdaBoostClassifier(customMLPClassifer(**params),
                                    n_estimators=n_estimators
                                    )

    train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)
    model_tune.fit(train_data, train_label)

    return model_tune

    # pred_proba = model_tune.predict_proba(test_data)[:, 1]
    # pred_y = model_tune.predict(test_data)
    #
    # prec, recall, false_alarm, auc_value, F_measure, G_measure, bal = calc_performance(test_label, pred_y, pred_proba)
    #
    # if metrics == 'F_measure':
    #     result = F_measure
    #
    # elif metrics == 'AUC_PR':
    #     result = auc_value
    #
    # elif metrics == 'G_measure':
    #     result = G_measure
    #
    # elif metrics == 'Bal_value':
    #     result = bal
    #
    # else:
    #     print('PLEASE SELECT THE METRICS !!!')
    #
    # return model_tune, result

