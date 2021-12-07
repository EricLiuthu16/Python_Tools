# -*- coding:utf-8 -*-
"""
@author: lzy <liuzhy.20@pbcsf.tsinghua.edu.cn>
@file: function_tool_box.py
@time:2021/12/05
@content:
    一些积累的功能性函数，目前主要包括：
        1）因子正交化矩阵计算
        2）根号，log带符号的缩尾
        3）自动填充的shift函数
        4）winsorize
        5）计算数据和均匀分布的KL散度
        6）IC（矩阵计算方法）
        7) Newy West T stats
"""

import numpy as np
import pandas as pd
import scipy.stats.entropy as entropy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def factor_neutralization(fac_mat, neu_mat):
    """
    因子正交化函数，one to one
    :param fac_mat: 因子矩阵(每一列是一个截面）
    :param neu_mat: 被正交化因子矩阵（每一列是一个截面）
    :return:
    """
    # check形态是否一致
    if fac_mat.shape != neu_mat.shape:
        raise print('因子矩阵维度不符，请检查！')

    new_fac = np.zeros_like(fac_mat)
    for col in range(fac_mat.shape[1]):
        try:
            y = fac_mat[:, col].reshape(-1, 1)
            x = np.hstack([np.ones_like(y).reshape(-1, 1),neu_mat[:, col].reshape(-1, 1)])

            #clean_nan
            na_idx = np.argwhere(~np.isnan(y))
            train_x = x[na_idx[:, 0], :]
            train_y = y[na_idx[:, 0], :]

            # regression
            beta = np.linalg.pinv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)
            res = y - x.dot(beta)
            new_fac[:, col] = res.reshape(-1)

        except Exception as e:
            print(e)
            new_fac[:, col] = np.nan
            print(f'problem occurred, fill nan at this section')
    return new_fac

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 带方向的开根号进行缩尾
def square_root_with_neg(x):
    x = np.float(x)
    return ((np.abs(x) + 1)**0.5 - 1) * np.sign(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 带方向的log进行缩尾
def log_with_neg(x):
    x = np.float(x)
    sig = np.sign(x)
    if sig >= 0:
        return np.log(1 + x)
    else:
        return -np.log(np.abs(-1 + x))  # x为负数的时候

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 带填充的shift函数
def shift_df(df,shift_periods,min_periods,axis=0):
    if shift_periods < 0:
        raise NotImplementedError

    select = np.full(df.shape,False)
    if axis == 0:
        # 第一行不能为na
        tofill_idx = ~df.iloc[0, :].isnull().values
        select[min_periods:shift_periods,tofill_idx] = True

    elif axis == 1:
        # 第一列不能为na，先找出第一列中不为0的行
        tofill_idx = ~df.iloc[:, 0].isnull().values
        select[tofill_idx,min_periods:shift_periods] = True

    shiftted = df.shift(shift_periods,axis=axis) # 正常shift
    bfill_shiftted = shiftted.fillna(method ='bfill',axis=axis) # 使用bfiil填充后的shift

    res = pd.DataFrame(data= np.where(select,bfill_shiftted,shiftted),index=df.index,columns=df.columns)
    return res

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def winsorize(arr):

    QT10 = np.nanquantile(arr, 0.05)
    QT90 = np.nanquantile(arr, 0.9)
    median = np.nanquantile(arr, 0.5)

    up = median + 2.5*(QT90 - QT10)
    down = median - 2.5*(QT90 - QT10)

    arr[arr >= up] = up
    arr[arr <= down] = down

    return arr

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_KL(arr, n):
    """
    计算输入arr数据分布和对应均匀分布的KL散度，区间数目为n
    :param arr:
    :return:
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception as e:
            print(e)
            print('input format wrong, plz check!')

    arr = winsorize(arr)
    slice = np.linspace(min(arr), max(arr), n+1)

    num_of_elements = []
    for idx, up in enumerate(slice[1:]):
        down = slice[idx - 1]
        if idx == len(slice) - 1:
            num_of_elements.append(len(arr[(arr>=down) & (arr<=up)]))
        else:
            num_of_elements.append(len(arr[(arr>=down) & (arr<up)]))
    p = np.array(num_of_elements) / len(arr)
    q = [1/n for i in range(n)]

    return entropy(p, q)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def cal_corr(
            X: np.ndarray,
            Y: np.ndarray
) -> np.ndarray:
    """
    计算行与行之间的corr值, 是一个nan robust函数
    :param X: mat 1
    :param Y: mat 2  
    :return: mat1 & mat2 行之间的corr，一般用来矩阵化计算IC
    """
    # 标准化
    x_ = X - np.nanmean(X, axis=1).reshape(-1, 1)
    y_ = Y - np.nanmean(Y, axis=1).reshape(-1, 1)

    # x,y任意一个为nan的位置，两个值都要为nan
    # cov
    x_cal = np.copy(x_)
    y_cal = np.copy(y_)
    x_cal[np.isnan(y_cal)] = np.nan
    y_cal[np.isnan(x_cal)] = np.nan

    x_cal[np.isnan(x_cal)] = 0.0
    y_cal[np.isnan(y_cal)] = 0.0

    cov = np.dot(x_cal, y_cal.T)

    # std
    std_x = np.nanstd(x_, axis=1).reshape(-1, 1) * np.ones_like(cov)
    std_y = np.nanstd(y_, axis=1).reshape(-1, 1) * np.ones_like(cov)
    std = np.dot(std_x, std_y.T)

    # corr
    n = np.sum(~np.isnan(x_cal), axis=1).reshape(-1, 1)
    corr = (cov / std) * (len(std) / n)

    return np.diag(corr)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calc_t_stat(beta, nw_lags_beta):
    """
    TODO 这是newy west t吗
    :param beta:
    :param nw_lags_beta:
    :return:
    """
    N = len(beta)
    B = beta - beta.mean(0)
    C = np.dot(B.T, B) / N

    if nw_lags_beta is not None:
        for i in range(nw_lags_beta + 1):
            cov = np.dot(B[i:].T, B[:(N - i)]) / N
            weight = i / (nw_lags_beta + 1)
            C += 2 * (1 - weight) * cov

    mean_beta = beta.mean(0)
    std_beta = np.sqrt(np.diag(C)) / np.sqrt(N)
    t_stat = mean_beta / std_beta

    return mean_beta, std_beta, t_stat

