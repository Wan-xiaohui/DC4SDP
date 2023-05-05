import os
import re
import numpy as np
from sklearn.model_selection import train_test_split

import collections
import copy
import json
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats
import seaborn as sns
import warnings

"""
Credit to https://github.com/janvanrijn/openml-pimp
"""


def fnameList(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            fnameList(file_path, list_name)
        else:
            list_name.append(file_path)


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


# def GetData(filename, showType=False):
#     if 'JURECZKO' in filename:
#         with open(filename, 'r') as f:
#             data = f.readlines()
#         x = []
#         y = []
#         empty = []
#
#         # get the types of metrics from first line
#         type = data[0].strip().split(',')
#         type.pop()
#         type.pop(0)
#
#         # get the detail data of metrics
#         for line in data[1:]:
#             tmp = []
#
#             odom = line.strip().split(',')
#
#             # delete the project information
#             for i in range(len(odom)):
#                 if is_number(odom[i]):
#                     tmp.append(float(odom[i]))
#                 else:
#                     if i not in empty:
#                         empty.append(i)
#                     tmp.append(0)
#
#             if tmp.pop() > 0:
#                 y.append(1)
#             else:
#                 # y.append(-1)
#                 y.append(0)
#
#             x.append(tmp)
#
#         x = np.delete(np.asarray(x), empty, axis=1)
#         empty = sorted(empty)
#         for i in range(len(empty)):
#             type.pop(empty[len(empty) - i - 1])
#
#         if showType:
#             return x, np.asarray(y), type
#         else:
#             return x, np.asarray(y)
#
#     else:
#         with open(filename, 'r') as f:
#             data = f.readlines()  # txt中所有字符串读入data
#             x = []
#             y = []
#             type = []
#
#             for line in data:
#                 if '###' in line:
#                     odom = line.strip().split(' ')
#                     odom.remove('###')
#                     type = odom
#                 else:
#                     tmp = []
#                     odom = line.strip().split(',')  # 将单个数据分隔开存好
#                     if not is_number(odom[0]):
#                         continue
#                     for item in odom:
#                         if is_number(item):
#                             tmp.append(float(item))
#                         elif (item == 'true') or (item == 'TRUE') or (item == 'Y') or (item == 'buggy'):
#                             y.append(1)
#                         else:
#                             y.append(0)
#                     x.append(tmp)
#
#         if showType:
#             return np.asanyarray(x), np.asarray(y), type
#         else:
#             return np.asanyarray(x), np.asarray(y)


def GetData(filename, showType=False):
    if 'PROMISE' in filename:
        with open(filename, 'r') as f:
            data = f.readlines()
        x = []
        y = []
        empty = []

        # get the types of metrics from first line
        type = data[0].strip().split(',')
        type.pop()

        # get the detail data of metrics
        for line in data[1:]:
            tmp = []

            odom = line.strip().split(',')

            # delete the project information
            for i in range(len(odom)):
                if is_number(odom[i]):
                    tmp.append(float(odom[i]))
                else:
                    if i not in empty:
                        empty.append(i)
                    tmp.append(0)

            if tmp.pop() > 0:
                y.append(1)
            else:
                # y.append(-1)
                y.append(0)

            x.append(tmp)

        x = np.delete(np.asarray(x), empty, axis=1)
        empty = sorted(empty)
        for i in range(len(empty)):
            type.pop(empty[len(empty) - i - 1])

        if showType:
            return x, np.asarray(y), type
        else:
            return x, np.asarray(y)

    elif 'JIT' in filename:
        with open(filename, 'r') as f:
            data = f.readlines()
        x = []
        y = []
        empty = []

        # get the types of metrics from first line
        type = data[0].strip().split(',')
        type.pop(0)
        type.pop(0)
        type.pop()

        # get the detail data of metrics
        for line in data[1:]:
            tmp = []

            odom = line.strip().split(',')

            # delete the project information
            for i in range(2, len(odom)):
                if is_number(odom[i]):
                    tmp.append(float(odom[i]))
                else:
                    if i not in empty:
                        empty.append(i)
                    tmp.append(0)

            if tmp.pop() > 0:
                y.append(1)
            else:
                # y.append(-1)
                y.append(0)

            x.append(tmp)

        x = np.delete(np.asarray(x), empty, axis=1)

        if showType:
            return x, np.asarray(y), type
        else:
            return x, np.asarray(y)

    elif 'ARB' in filename:
        with open(filename, 'r') as f:
            data = f.readlines()  # txt中所有字符串读入data
            x = []
            type = []

            for line in data:
                if '###' in line:
                    odom = line.strip().split(' ')
                    odom.remove('###')
                    type = odom
                else:
                    tmp = []
                    odom = line.strip().split(',')  # 将单个数据分隔开存好
                    if (not is_number(odom[0])) and len(odom) == 1:
                        continue

                    for item in odom:
                        item = item.strip()
                        if is_number(item):
                            tmp.append(float(item))
                        # elif (item == 'true') or (item == 'TRUE') or (item == 'Y') or (item == 'buggy'):
                        #     y.append(1)
                        # else:
                        #     y.append(0)
                    x.append(tmp)

            y = [int(x[i].pop()) for i in range(len(x))]

        if showType:
            return x, np.asarray(y), type
        else:
            return x, np.asarray(y)

    else:
        with open(filename, 'r') as f:
            data = f.readlines()  # txt中所有字符串读入data
            x = []
            y = []
            type = []

            for line in data:
                if '###' in line:
                    odom = line.strip().split(' ')
                    odom.remove('###')
                    type = odom
                else:
                    tmp = []
                    odom = line.strip().split(',')  # 将单个数据分隔开存好
                    if not is_number(odom[0]):
                        continue

                    for item in odom:
                        if is_number(item):
                            tmp.append(float(item))
                        elif (item == 'true') or (item == 'TRUE') or (item == 'True') \
                                or (item == 'Y') or (item == 'buggy'):
                            y.append(1)
                        else:
                            y.append(0)
                    x.append(tmp)

        if 'Eclipse34_debug' in filename:
            # 该数据集中倒数第2列为缺陷数量，该数值不应该在训练过程中出现，故而删去该列
            for i in range(len(x)):
                x[i].pop(-2)

        if showType:
            return np.asanyarray(x), np.asarray(y), type

        else:
            return np.asanyarray(x), np.asarray(y)


def load_data(folder_path, return_cols=False):

    begin_num = 1
    end_num = 36

    flist = []
    fname = []
    group = sorted(['AEEEM', 'NASA', 'PROMISE', 'ReLink'])
    data_list, label_list = [], []
    metric_list = []

    for i in range(len(group)):
        tmp = []
        fnameList(folder_path + group[i], tmp)
        tmp = sorted(tmp)
        flist.append(tmp)

    for c in range(begin_num, end_num + 1):
        if c in range(6):
            tmp = flist[0].copy()
            target = tmp.pop(c - 1)
            fname.append(target.split('/')[2][:-5])
            # fname.append(target.split('\\')[1][:-5])

        if c in range(6, 18):
            tmp = flist[1].copy()
            target = tmp.pop(c - 6)
            fname.append(target.split('/')[2][:-5])
            # fname.append(target.split('\\')[1][:-5])

        # if c in range(9, 13):
        #     tmp = flist[2].copy()
        #     target = tmp.pop(c - 9)
        #     # fname.append(target.split('/')[2][:-5])
        #     fname.append(target.split('\\')[1][:-5])

        if c in range(18, 33):
            tmp = flist[2].copy()
            target = tmp.pop(c - 18)
            fname.append(target.split('/')[2][:-4])
            # fname.append(target.split('\\')[1][:-4])

        if c in range(33, 37):
            tmp = flist[3].copy()
            target = tmp.pop(c - 33)
            fname.append(target.split('/')[2][:-5])
            # fname.append(target.split('\\')[1][:-5])

        # if c in range(27, 42):
        #     tmp = flist[4].copy()
        #     target = tmp.pop(c - 27)
        #     fname.append(target.split('/')[2][:-4])
        #     # fname.append(target.split('\\')[1][:-4])
        #
        # if c in range(42, 46):
        #     tmp = flist[5].copy()
        #     target = tmp.pop(c - 42)
        #     fname.append(target.split('/')[2][:-5])
        #     # fname.append(target.split('\\')[1][:-5])

        if not return_cols:
            x, y = GetData(target, False)
            data = np.array(x)
            label = np.array(y)
            metric = []

            # data_list.append(data)
            # label_list.append(label)

        else:
            x, y, metric = GetData(target, True)
            data = np.array(x)
            label = np.array(y)
            # print(metric)

        data_list.append(data)
        label_list.append(label)
        metric_list.append(metric)

        # data = np.array(x)
        # label = np.array(y)
        #
        # data_list.append(data)
        # label_list.append(label)

    if not return_cols:
        return data_list, label_list, fname

    else:
        return data_list, label_list, fname, metric_list


def shuffle_data(data, label):
    indices = np.random.permutation(data.shape[0])
    data_r = data[indices, :]
    label_r = label[indices]
    return data_r, label_r


def SfindCommonMetric(fsource, ftarget, showDiff=False, showType=False):
    sx, sy, Stype = GetData(fsource, showType=True)
    tx, ty, Ttype = GetData(ftarget, showType=True)

    common = []

    ss = sx.shape
    tt = tx.shape
    for i in range(ss[1]):
        if Stype[i] in Ttype:
            common.append(Stype[i])

    if len(common) > 0:
        fsx = np.zeros((ss[0], len(common)))
        ftx = np.zeros((tt[0], len(common)))
        for i in range(len(common)):
            index = Stype.index(common[i])
            fsx[:, i] = sx[:, index]

            index = Ttype.index(common[i])
            ftx[:, i] = tx[:, index]

        DiffSx = np.zeros((ss[0], ss[1] - len(common)))
        DiffTx = np.zeros((tt[0], tt[1] - len(common)))

        i = 0
        for j in range(ss[1]):
            if Stype[j] not in common:
                DiffSx[:, i] = sx[:, j]
                i = i + 1
        i = 0
        for j in range(tt[1]):
            if Ttype[j] not in common:
                DiffTx[:, i] = tx[:, j]
                i = i + 1

        if showDiff and showType:
            return fsx, sy, ftx, ty, DiffSx, DiffTx, common

        elif showDiff and (not showType):
            return fsx, sy, ftx, ty, DiffSx, DiffTx

        elif (not showDiff) and showType:
            return fsx, sy, ftx, ty, common

        else:
            return fsx, sy, ftx, ty
    else:
        return 0, 0, 0, 0


def MfindCommonMetric(list, ftarget, split=False):
    tx, ty, Ttype = GetData(ftarget, showType=True)
    tt = tx.shape
    common = []

    flist = list.copy()
    ### find the common metric
    first = 1
    dump = []

    for item in flist:
        x, y, Stype = GetData(item, showType=True)
        ss = x.shape

        if first == 1:
            for i in range(ss[1]):
                if Stype[i] in Ttype:
                    common.append(Stype[i])
            first = 0
        else:
            for i in range(len(common)):
                if common[i] not in Stype and i not in dump:
                    dump.append(i)
    dump = sorted(dump, reverse=True)
    for i in range(len(dump)):
        common.pop(dump[i])

    ### read the data and concatendate

    if len(common) == 0:
        return 0, 0, 0, 0, []
    else:
        ftx = np.zeros((tt[0], len(common)))
        for i in range(len(common)):
            index = Ttype.index(common[i])
            ftx[:, i] = tx[:, index]

        sx, sy, Stype = GetData(flist.pop(), showType=True)

        fsx = np.zeros((len(sy), len(common)))
        for i in range(len(common)):
            index = Stype.index(common[i])
            fsx[:, i] = sx[:, index]

        loc = []
        base = 0

        for item in list:
            x, y, Type = GetData(item, showType=True)
            loc.append(base)
            base += len(y)
            fx = np.zeros((len(y), len(common)))
            for i in range(len(common)):
                index = Type.index(common[i])
                fx[:, i] = x[:, index]
            fsx = np.concatenate((fsx, fx), axis=0)
            sy = np.concatenate((sy, y), axis=0)

        if split:
            return fsx, sy, ftx, ty, loc
        else:
            return fsx, sy, ftx, ty, []


def GetDataList(flist):
    a = flist.pop()
    xs, ys, xt, yt, loc = MfindCommonMetric(flist, a)
    x = np.concatenate((xs, xt), axis=0)
    y = np.concatenate((ys, yt), axis=0)
    return x, y


def collectData(fname):
    count = len(open(fname, 'r').readlines())
    with open(fname, 'r') as f:
        tmp = list(map(eval, f.readline()[1:-2].split()))
        res = np.zeros((count - 1, len(tmp)))
        i = 0
        print(fname, len(tmp))
        for line in f:
            line = line[1:-2]
            res[i] = np.asarray(line.split())[:len(tmp)]
            i += 1
            print(np.asarray(line.split()))
    return np.concatenate(([tmp], res))


def resCollect():
    for mode in {'all', 'clf', 'seq', 'adpt'}:
        resDir = 'res' + mode.upper()
        for root, dirs, files in os.walk(resDir):
            for name in files:
                fname = os.path.join(root, name)
                result = collectData(fname)
                np.savetxt(fname, result, fmt='%.7f')


def normal(xs, xt):
    ss = xs.shape
    tt = xt.shape

    # normalization for source data
    res = np.zeros((ss[0], ss[1]))
    for i in range(ss[1]):
        tmp = xs[:, i]
        minm = np.min(tmp)
        maxm = np.max(tmp)
        res[:, i] = (tmp - minm) / (maxm - minm)
    xs = res

    # normalization for target data
    res = np.zeros((tt[0], tt[1]))
    for i in range(tt[1]):
        tmp = xt[:, i]
        minm = np.min(tmp)
        maxm = np.max(tmp)
        res[:, i] = (tmp - minm) / (maxm - minm)
    xt = res

    return xs, xt


def rank_dict(dictionary, reverse=False):
    '''
    Get a dictionary and return a rank dictionary
    for example dic={'a':10,'b':2,'c':6}
    will return dic={'a':1.0,'b':3.0,'c':2.0}

    '''
    dictionary = copy.copy(dictionary)

    if reverse:

        for key in dictionary.keys():
            dictionary[key] = 1 - dictionary[key]

    sortdict = collections.OrderedDict(sorted(dictionary.items()))
    ranks = scipy.stats.rankdata(list(sortdict.values()))
    result = {}

    for idx, (key, value) in enumerate(sortdict.items()):
        result[key] = ranks[idx]

    return result


def sum_dict_values(a, b, allow_subsets=False):
    '''
    Get two dictionary sum them together!
    '''
    result = {}
    a_total = sum(a.values())
    b_total = sum(b.values())
    a_min_b = set(a.keys()) - set(b.keys())
    b_min_a = set(b.keys()) - set(a.keys())

    #     if len(b_min_a) > 0:
    #         raise ValueError('dict b got illegal keys: %s' %str(b_min_a))

    #     if not allow_subsets and len(a_min_b):
    #         raise ValueError('keys not the same')

    for idx in a.keys():
        if idx in b:
            result[idx] = a[idx] + b[idx]
        else:
            result[idx] = a[idx]

    #     if sum(result.values()) != a_total + b_total:
    #         raise ValueError()

    return result


def obtain_marginal_contributions(df):
    '''
    This is the main function that calls Top functions
    '''

    all_ranks = dict()
    all_tasks = list()
    total_ranks = None
    num_tasks = 0
    marginal_contribution = collections.defaultdict(list)

    lst_datasets = list(df.dataset.unique())

    for dataset in lst_datasets:

        a = df[df.dataset == dataset]
        a = a.drop("dataset", axis=1)
        param = dict()

        for index, row in a.iterrows():
            marginal_contribution[row["param"]].append(row["importance"])
            param.update({row["param"]: row["importance"]})

        ranks = rank_dict(param, reverse=True)
        if total_ranks is None:
            total_ranks = ranks
        else:
            total_ranks = sum_dict_values(ranks, total_ranks, allow_subsets=False)
            num_tasks += 1
    total_ranks = divide_dict_values(total_ranks, num_tasks)
    return total_ranks, marginal_contribution, lst_datasets


def marginal_plots(sorted_values, keys, fig_title):
    sorted_values = sorted_values[0:8]
    keys = keys[0:8]
    plt.figure(figsize=(12, 10))
    plt.violinplot(list(sorted_values), list(range(len(sorted_values))), showmeans=True)
    plt.plot([-0.5, len(sorted_values) - 0.5], [0, 0], 'k-', linestyle='--', lw=1)
    keys = [format_name(key) for key in keys]
    plt.xticks(list(range(len(sorted_values))), list(keys), rotation=45, ha='right')
    plt.ylabel('marginal contribution')
    plt.title(fig_title)
    plt.show()
    plt.savefig("results/plots/" + fig_title + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()


def format_name(name):
    '''
    Format hyperparameter names!
    '''
    mapping_plain = {
        'strategy': 'imputation',
        'max_features': 'max. features',
        'min_samples_leaf': 'min. samples leaf',
        'min_samples_split': 'min. samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning rate',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algorithm',
    }

    mapping_short = {
        'strategy': 'imputation',
        'max_features': 'max. feat.',
        'min_samples_leaf': 'samples leaf',
        'min_samples_split': 'samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning r.',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algo.',
    }

    parts = name.split('__')

    for idx, part in enumerate(parts):
        if part in mapping_plain:
            if len(parts) < 3:
                parts[idx] = mapping_plain[part]
            else:
                parts[idx] = mapping_short[part]

    return ' / '.join(parts)


def divide_dict_values(d, denominator):
    '''
    divide d/demoniator
    '''
    result = {}

    for idx in d.keys():
        result[idx] = d[idx] / denominator

    return result


def determine_relevant(data, max_items=None, max_interactions=None):
    sorted_values = []
    keys = []
    interactions_seen = 0

    for key in sorted(data, key=lambda k: median(data[k]), reverse=True):
        if '__' in key:
            interactions_seen += 1
            if interactions_seen > max_interactions:
                continue

        sorted_values.append(data[key])
        keys.append(key)

    if max_items is not None:
        sorted_values = sorted_values[:max_items]
        keys = keys[:max_items]

    return sorted_values, keys


def cls_kde_plot(file_path, cls, metric, important_hyperparameter, x1, x2, y1, y2, b=0, kernel=None, scale=None):
    # file_path="../PerformanceData/total/AB_results_total.csv"
    df = pd.read_csv(file_path)
    df_total = pd.DataFrame()
    col_name = "CV_" + metric
    for item in df.dataset.unique():
        df_dataset = df.loc[df['dataset'] == item]
        # max_auc=max(df_dataset["CV_auc"])
        df_row = df_dataset.loc[df_dataset[col_name] == max(df_dataset[col_name])]
        df_total = df_total.append(df_row)

    if kernel != None:
        df_total = df_total[df_total[important_hyperparameter] == kernel]
        important_hyperparameter = "gamma"

    plt.figure(figsize=(7, 9))

    # set bandwidth for kde
    if b != 0:
        sns.kdeplot(df_total[important_hyperparameter], bw=b)
    else:
        sns.kdeplot(df_total[important_hyperparameter])

    if kernel != None:
        plt_title = cls + "-" + kernel + ":" + important_hyperparameter
        plt.title(plt_title)
    else:
        plt_title = cls + ":" + important_hyperparameter
        plt.title(plt_title)

    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    if scale != None:
        plt.xscale(scale)
    plt.savefig("../results/plots/" + plt_title + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
