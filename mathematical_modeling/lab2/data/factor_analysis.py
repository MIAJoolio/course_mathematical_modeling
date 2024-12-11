import os
import glob
from typing import Tuple, Literal
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

class LinearRegressionAnalysis:
    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Параметры класса:
        data: данные с таргетом
        target_col: название колонки с таргетом
        """
        self.data = data
        self.target_col = target_col
        self.X = data.drop(target_col, axis=1)
        self.y = data[target_col]
        self.model = None
    
    def feature_analysis(self):
        # Вычисляем корреляционную матрицу для всего DataFrame (включая таргет)
        corr_matrix = self.data.corr()

        # Создаем список для результатов для X
        results_X = []
        for feature in self.X.columns:
            # Мультиколлинеарность для X: проверка на пороговое значение корреляции с другими предикторами
            multicollinearity_X = any(abs(corr_matrix[feature].drop(feature)) > 0.7)  # исключаем само-сравнение
            
            # Теснота для X: корреляция с таргетом
            target_correlation_X = corr_matrix[feature][self.target_col]  # Исправлено на использование self.target_col
            tightness_X = abs(target_correlation_X) < 0.01  # если корреляция с таргетом меньше 0.01
            
            # Отклик для X: определяем знак корреляции с таргетом
            response_X = None
            if target_correlation_X > 0:
                response_X = "+"
            elif target_correlation_X < 0:
                response_X = "-"
        
            # Добавляем информацию в результат
            results_X.append({
                "feature": feature,
                "multicollinearity": multicollinearity_X,
                "tightness": tightness_X,
                "response": response_X
            })
        
        # Результаты только для таргета в df
        target_results = []
        
        # Мультиколлинеарность для таргета: проверка на пороговое значение корреляции с предикторами
        for feature in self.X.columns:
            multicollinearity_df = abs(corr_matrix[self.target_col][feature]) > 0.7  # Исправлено на использование self.target_col
            target_results.append({
                "feature": f"target_{feature}",  # Формируем строку вида target_X1, target_X2 и т.д.
                "multicollinearity": multicollinearity_df,
                "tightness": None,  # Таргет не может иметь тесноту, так как это отклик
                "response": None
            })
        
        # Преобразуем список в DataFrame для X
        result_df_X = pd.DataFrame(results_X)
        result_df_df = pd.DataFrame(target_results)
        
        true_labels = result_df_X.loc[(result_df_X.multicollinearity == False) & (result_df_X.tightness == False) & (result_df_X.response == '+'), 'feature'].values

        # Вернем два результата и одну матрицу корреляции
        return result_df_X, result_df_df, corr_matrix, true_labels


    def estimate_linear_regression(self, selected_factors:list=None):
        """
        Оцените модель линейной регрессии

        Параметры:
        selected_factors: конкретные факторы для включения в модель. По умолчанию используются все факторы

        Returns:
        sm.OLS: обученная линейная модель
        """
        if selected_factors is None:
            selected_factors = self.X.columns
        X_selected = self.X[selected_factors]
        X_selected = sm.add_constant(X_selected)
        self.model = sm.OLS(self.y, X_selected).fit()
        return self.model

    def evaluate_model(self):
        """
        Функция оценки эффективности модели линейной регрессии

        Returns:
        dict: словарь с метриками MSE, RMSE, relative error
        """
        if self.model is None:
            raise ValueError("No model has been estimated. Run estimate_linear_regression() first.")

        predictions = self.model.predict(sm.add_constant(self.X[self.model.model.exog_names[1:]]))
        mse = mean_squared_error(self.y, predictions)
        rmse = np.sqrt(mse)
        n = len(self.y)
        relative_error = (1 / n) * np.sum(np.abs((self.y - predictions) / self.y))

        # Статистическая оценка
        f_statistic = self.model.fvalue  # Критерий Фишера
        r_squared = self.model.rsquared  # Коэффициент детерминации

        # Оценка критериями Стьюдента для каждого коэффициента
        student_t_test = self.model.tvalues  # Критерий Стьюдента для каждого коэффициента

        return {
            'MSE': mse,
            'RMSE': rmse.item(),
            'Relative Error': relative_error.item(),
            'F-statistic': f_statistic.item(),
            'R-squared': r_squared.item(),
            'Student t-test': student_t_test.tolist()  # Возвращаем все t-статистики
        }

    def add_lag_to_feature(self, feature_name: str, max_lag: int):
        """
         Функция добавления лагов к некоторой переменной x_i

        Parameters:
        feature_name: название переменной x_i
        max_lag: максимальный лаг

        Returns:
        pd.DataFrame: Датафрейм с новыми лаггированными переменными
        """
        for lag in range(1, max_lag + 1):
            self.data[f'{feature_name}_lag{lag}'] = self.data[feature_name].shift(lag)
        
        # Удаляем строки с NaN после сдвигов
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Обновляем X и y
        self.X = self.data.drop(self.target_col, axis=1)
        self.y = self.data[self.target_col]
        
        return self.data

    def get_summary(self):
        """
        Get the summary of the fitted model.

        Returns:
        str: A textual summary of the regression model.
        """
        if self.model is None:
            raise ValueError("No model has been estimated. Run estimate_linear_regression() first.")
        return self.model.summary().as_text()

    def predict(self, new_data):
        """
        Predict target variable using the fitted model.

        Parameters:
        new_data (pd.DataFrame): New data for prediction, containing the same factors as the model.

        Returns:
        np.array: Predicted target values.
        """
        if self.model is None:
            raise ValueError("No model has been estimated. Run estimate_linear_regression() first.")
        selected_factors = self.model.model.exog_names[1:]  # Исключаем константу
        X_new = sm.add_constant(new_data[selected_factors], has_constant='add')
        return self.model.predict(X_new)

    def extract_consecutive_range(self, sb: int, se: int):
        """
        Выделить подотрезок исходного ряда отклика y для индексов от sb до se.

        Параметры:
        sb: Начальный индекс подотрезка.
        se: Конечный индекс подотрезка.

        Возвращает:
        pd.DataFrame: Новый датафрейм с подотрезком данных и обновленным y.
        """
        # Проверка, что подотрезок содержит хотя бы 5-8 наблюдений
        if se - sb < 5:  # Проверка на минимальную длину
            raise ValueError("The selected range is too small. It should contain at least 5-8 observations.")
        
        # Извлекаем подотрезок данных
        subset_data = self.data.iloc[sb-1:se]
        
        # Обновляем y для подотрезка
        subset_X = subset_data.drop(self.target_col, axis=1)
        subset_y = subset_data[self.target_col]
        
        # Обновляем X и y в объекте
        self.X = subset_X
        self.y = subset_y
        
        return subset_data

    

if __name__ == "__main__":
    # Пример использования
    data_dict = {
        'X1': np.random.randn(100)-10,
        'X2': np.random.randn(100)+231,
        'X3': np.random.randn(100),
        'X4': np.random.randn(100)+20,
        'X5': np.random.randn(100)-129,
        'target': np.random.randn(100)
    }

    # Создаем DataFrame с данными
    data = pd.DataFrame(data_dict)

    # Инициализация анализа
    analysis = LinearRegressionAnalysis(data, target_col='target')

    # Выполнение анализа на мультиколлинеарность
    print("Multicollinearity check:")
    result_df_X, result_df_df, corr_matrix, true_labels = analysis.feature_analysis()
    print(result_df_X)

    # Добавление лагов к переменной (например, X1)
    data_with_lags_X1 = analysis.add_lag_to_feature(feature_name='X1', max_lag=2)
    print("\nData with Lags for X1:")
    print(data_with_lags_X1.head())

    # Извлечение подотрезка из ряда отклика (например, с 20 по 40)
    data_subset = analysis.extract_consecutive_range(sb=20, se=60)
    print("\nExtracted data subset:")
    print(data_subset.head())

    # Оценка и построение модели на подотрезке данных
    selected_factors = true_labels  # Выбираем только те факторы, которые прошли проверку
    model = analysis.estimate_linear_regression(selected_factors=selected_factors)
    print("\nModel Summary:")
    print(analysis.get_summary())

    # Оценка модели
    print("\nModel Evaluation:")
    evaluation_metrics = analysis.evaluate_model()
    print(evaluation_metrics)

    # Пример предсказания для новых данных
    new_data = pd.DataFrame({
        factor: [np.mean(data[factor])] for factor in selected_factors
    })
    print("\nPrediction for new data:")
    print(analysis.predict(new_data))
