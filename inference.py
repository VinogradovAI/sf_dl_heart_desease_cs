import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from typing import List, Tuple, Dict

# --------------------
# Constants and Configurations
# --------------------

# File paths
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'data/submission.csv'
README_PATH = 'data/README.txt'

# Paths to saved objects
SCALER_PATH = os.path.join('models', 'scaler.pkl')
PCA_PATH = os.path.join('models', 'pca.pkl')
ENCODERS_PATH = os.path.join('models', 'encoders.pkl')
MODEL_PATH = os.path.join('models', 'mlp2.pth')

# Columns for encoding
ONE_HOT_COLUMNS = ['Sex', 'Resting_electrocardiographic_results']
LABEL_ENCODE_COLUMNS = ['Number_of_major_vessels', 'Thal', 'Chest_bin', 'Slope']

# PCA settings
PCA_COMPONENTS = 5

# Neural Network settings
MLP2_HIDDEN_LAYERS = [64, 128, 64, 32]

# --------------------
# Define MLP2 Model
# --------------------

class MLP2(nn.Module):
    """
    Гибкая многослойная перцептронная модель с переменным числом скрытых слоев.
    """
    def __init__(self, input_size: int, hidden_layers: list):
        super(MLP2, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --------------------
# Utility Functions
# --------------------

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

def preprocess_data(df: pd.DataFrame, scaler: StandardScaler, pca: PCA, encoders: dict) -> np.ndarray:
    """
    Применение предобработки: заполнение пропусков, кодирование, масштабирование и PCA.

    Args:
        df (pd.DataFrame): DataFrame для предобработки.
        scaler (StandardScaler): Обученный StandardScaler.
        pca (PCA): Обученный PCA.
        encoders (dict): Словарь с обученными LabelEncoders.

    Returns:
        np.ndarray: Предобработанные и преобразованные признаки.
    """
    # Заполнение пропусков
    df = fill_missing_values(df)

    # Кодирование категориальных переменных
    df = selective_encode(df, ONE_HOT_COLUMNS, LABEL_ENCODE_COLUMNS, encoders)

    # Масштабирование данных
    X_scaled = scaler.transform(df)

    # Применение PCA
    X_pca = pca.transform(X_scaled)

    return X_pca

def load_preprocessing_objects(scaler_path: str, pca_path: str, encoders_path: str) -> tuple:
    """
    Загрузка scaler, PCA и энкодеров с диска.

    Args:
        scaler_path (str): Путь к файлу scaler.pkl.
        pca_path (str): Путь к файлу pca.pkl.
        encoders_path (str): Путь к файлу encoders.pkl.

    Returns:
        tuple: scaler, pca, encoders
    """
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    encoders = joblib.load(encoders_path)
    return scaler, pca, encoders

def load_model(model_path: str, input_size: int, hidden_layers: list) -> nn.Module:
    """
    Загрузка обученной нейронной сети.

    Args:
        model_path (str): Путь к файлу модели mlp2.pth.
        input_size (int): Количество входных признаков.
        hidden_layers (list): Список размеров скрытых слоев.

    Returns:
        nn.Module: Загруженная модель.
    """
    model = MLP2(input_size=input_size, hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_predictions(predictions: np.ndarray, filename: str = SUBMISSION_PATH) -> None:
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

def write_results_to_file(text: str, filename: str = README_PATH) -> None:
    """
    Добавление текста в файл README.txt.

    Args:
        text (str): Текст для записи.
        filename (str): Путь к файлу.
    """
    with open(filename, 'a') as file:
        file.write(text + '\n')

# --------------------
# Main Inference Function
# --------------------

def main():
    """
    Основная функция для выполнения инференса:
    загрузка предобработчиков и модели, загрузка и предобработка тестовых данных,
    выполнение предсказаний и сохранение результатов.
    """
    # 1. Загрузка предобработчиков
    try:
        scaler, pca, encoders = load_preprocessing_objects(SCALER_PATH, PCA_PATH, ENCODERS_PATH)
        success_message = "Предобработчики успешно загружены."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при загрузке предобработчиков: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

    # 2. Загрузка модели
    try:
        input_size = PCA_COMPONENTS
        model = load_model(MODEL_PATH, input_size=input_size, hidden_layers=MLP2_HIDDEN_LAYERS)
        success_message = "Модель успешно загружена."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при загрузке модели: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

# 3. Загрузка тестовых данных
    try:
        test_df = load_test_data(TEST_PATH)
        success_message = "Тестовые данные успешно загружены."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при загрузке тестовых данных: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

    # 4. Предобработка тестовых данных
    try:
        X_test_pca = preprocess_data(test_df, scaler, pca, encoders)
        success_message = "Предобработка тестовых данных завершена."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при предобработке тестовых данных: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

    # 5. Конвертация в тензор
    try:
        X_test_tensor = torch.FloatTensor(X_test_pca)
        success_message = "Данные конвертированы в тензор."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при конвертации данных в тензор: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

    # 6. Инференс
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = (outputs.numpy().flatten() >= 0.5).astype(int)
        success_message = "Инференс выполнен успешно."
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при выполнении инференса: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

    # 7. Сохранение предсказаний
    try:
        save_predictions(predictions, SUBMISSION_PATH)
        success_message = f"Предсказания сохранены в {SUBMISSION_PATH}"
        print(success_message)
        write_results_to_file(success_message, README_PATH)
    except Exception as e:
        error_message = f"Ошибка при сохранении предсказаний: {e}"
        print(error_message)
        write_results_to_file(error_message, README_PATH)
        return

if __name__ == '__main__':
    main()