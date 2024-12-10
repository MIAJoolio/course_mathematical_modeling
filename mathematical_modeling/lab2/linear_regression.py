import os
import glob
from typing import Tuple, Literal
import argparse

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt


def estimate_linear_regression(X,y):
    # добавляем константу
    X = sm.add_constant(X)
    # вычисляем модель через OLS
    model = sm.OLS(y, X).fit()
    return model
    # print(model.summary())


def main():

    print("if you don't what parameters to path type: python lab2/linear_regression.py --help")
    # lab2/data/final_dataset.txt
    
    # запускаем парсинг параметров для загрузки данных 
    parser = argparse.ArgumentParser(description='pass data to dataloader and linear regression')
    parser.add_argument('filepath', type=str, help='Input filepath')
    parser.add_argument('target_col', type=str, help='Input target column name')
    args = parser.parse_args()

    # 
    raw_data = pd.read_csv(filepath_or_buffer=args.filepath, sep=';')
    linear_regr1 = estimate_linear_regression(raw_data.drop(args.target_col, axis=1), raw_data[args.target_col])
    print(linear_regr1.summary())


if __name__ == '__main__':
    # python lab2/linear_regression.py lab2/data/final_dataset.txt growth_total 
    main()