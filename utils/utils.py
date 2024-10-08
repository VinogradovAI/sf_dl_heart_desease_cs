# utils/utils.py

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder


def load_test_data(test_path: str) -> pd.DataFrame:
    """
    Загрузка тестового набора данных.

    Args:
        test_path (str): Путь к тестовым данным.

    Returns:
        pd.DataFrame: DataFrame с тестовыми данными.
    """
    test_df = pd.read_csv(test_path)
    return test_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполнение пропущенных значений в DataFrame.

    Args:
        df (pd.DataFrame): DataFrame для обработки.

    Returns:
        pd.DataFrame: DataFrame с заполненными пропущенными значениями.
    """
    if 'id' in df.columns:
        df = df.drop(columns='id')

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df


def selective_encode(
        df: pd.DataFrame,
        one_hot_columns: List[str],
        label_encode_columns: List[str],
        encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Применение one-hot и label encoding к указанным столбцам.
    Args:
        df (pd.DataFrame): DataFrame для кодирования.
        one_hot_columns (List[str]): Список колонок для one-hot кодирования.
        label_encode_columns (List[str]): Список колонок для label кодирования.
        encoders (Dict[str, LabelEncoder]): Словарь с обученными LabelEncoders, ключи — названия колонок.
    Returns:
        pd.DataFrame: DataFrame с закодированными столбцами.
    """
    # One-Hot Encoding
    if one_hot_columns:
        df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)
        print(f"One-Hot Encoding применен к колонкам: {one_hot_columns}")

    # Label Encoding
    for col in label_encode_columns:
        if col in df.columns:
            if col in encoders:
                le = encoders[col]
                # Преобразуем только те значения, которые известны энкодеру
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                print(f"Label Encoding применен к колонке: {col}")
            else:
                print(f"Предупреждение: Нет обученного LabelEncoder для колонки '{col}'. Колонка пропущена.")
        else:
            print(f"Предупреждение: Колонка '{col}' не найдена в DataFrame.")

    return df



def save_predictions(predictions: np.ndarray, filename: str) -> None:
    """
    Сохранение предсказаний в CSV файл.

    Args:
        predictions (np.ndarray): Массив предсказаний.
        filename (str): Путь к файлу для сохранения.
    """
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'class': predictions.astype(int)
    })
    submission.to_csv(filename, index=False)


def write_results_to_file(text: str, filename: str) -> None:
    """
    Добавление текста в файл README.txt.

    Args:
        text (str): Текст для записи.
        filename (str): Путь к файлу.
    """
    with open(filename, 'a') as file:
        file.write(text + '\n')
