import os
import glob
from typing import Tuple, Literal
import argparse

import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import pearsonr
from itertools import combinations

#### 

#### 

def check_multicollinearity(X, threshold=0.7):

    correlation_matrix = X.corr()
    highly_correlated = [
        (i, j, correlation_matrix.loc[i, j]) 
        for i, j in combinations(correlation_matrix.columns, 2)
        if abs(correlation_matrix.loc[i, j]) > threshold
    ]
    return pd.DataFrame(highly_correlated, columns=['factor_1','factor_2','corr_value'])

def check_correlation_with_response(X, Y, threshold=0.3):
    correlations = {col: pearsonr(X[col], Y)[0] for col in X.columns}
    significant_factors = [col for col, r in correlations.items() if abs(r) > threshold]
    return significant_factors, correlations




def main():

    # print("if you don't what parameters to path type: python lab2/linear_regression.py --help")
    # lab2/data/final_dataset.txt
    
    # запускаем парсинг параметров для загрузки данных 
    parser = argparse.ArgumentParser(description='pass data to dataloader and linear regression')
    parser.add_argument('filepath', type=str, help='Input filepath')
    parser.add_argument('target_col', type=str, help='Input target column name')
    parser.add_argument('multicollinearity_threshold', type=float, help='Input multicollinearity threshold from 0 to 1')
    parser.add_argument('correlation_threshold', type=float, help='Input correlation threshold from 0 to 1')
    args = parser.parse_args()
     
    # filepath = 'lab2/data/final_dataset.txt'
    raw_data = pd.read_csv(filepath_or_buffer=args.filepath, sep=';')
    print("factors that will cause multicollinearity")
    print(check_multicollinearity(raw_data, args.multicollinearity_threshold))
    print('\n')

    significant_factors_with_response, correlations = check_correlation_with_response(raw_data.drop(args.target_col, axis=1), raw_data[args.target_col], args.correlation_threshold)
    print("Факторы, значимо связанные с откликом:", significant_factors_with_response)  
    print("Корреляция факторов с откликом:\n", correlations)
    print('\n')

if __name__ == '__main__':
    main()

# python lab2/factor_analys.py lab2/data/final_dataset.txt growth_total 0.7 0.1