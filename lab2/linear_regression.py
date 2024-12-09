import os
import glob
from typing import Tuple, Literal

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt



# def checker(true_cond:list=['y', 'Y', 'yes', True], false_cond:list=['n', 'N', 'no', False]):
#     answer = input()
#     if answer in true_cond:
#         return True
#     if answer in false_cond:
#         return False
# 
# 
# def load_data(file_path:str, files_ext:Literal['csv','xlsx']):
#     if files_ext == 'csv':
#         new_data = pd.read_csv(file_path)
#     elif files_ext == 'xlsx':
#         new_data = pd.read_excel(file_path)
    
#     print(new_data.head())
#     print('data ok?')
    
#     if checker():
#         return new_data
#     else:
#         if checker(true_cond=['rename']):
#             new_data = new_data.rename({})
# 
# 
# def create_panel_data(dir_path:str='lab2/data', files_ext:str='csv', keys_on_merge:list=['date']):
#     files = glob.glob(os.path.join(dir_path, f"*.{files_ext}"))
    
#     if files != []:
#         for i, file in enumerate(files):
#             if files_ext == 'csv':
#                 new_data = pd.read_csv(file)
#             elif files_ext == 'xlsx':
#                 new_data = pd.read_excel(file)
            
#             if i == 0:
#                 data_raw = new_data.copy()
#             else:
#                 data_raw = data_raw.merge(new_data, how='left', on=keys_on_merge)
# 
#     return data_raw





def main():
    print(os.getcwd())
    # load_data('lab2/data/23230000100010200001_Естественный_прирост_населения_за_год.xlsx', 'xlsx')
    # raw_df = create_panel_data(files_ext='xlsx', keys_on_merge=['keys_on_merge'])
    # print(raw_df.head())

if __name__ == '__main__':
    main()