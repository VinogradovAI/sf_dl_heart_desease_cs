# main.py

import warnings

warnings.simplefilter('ignore')  # Игнорируем предупреждения

import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Импортируем константы из config.py
from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, PLOTS_DIR, README_PATH,
    ONE_HOT_COLUMNS, LABEL_ENCODE_COLUMNS, NUMERIC_COLUMNS, OUTLIER_COLUMNS, TARGET_COLUMN, SCATTER_PLOT_COLUMNS, MLP1_HIDDEN_LAYERS,
    MLP2_HIDDEN_LAYERS, MLP_LEARNING_RATE, MLP_EPOCHS, MLP_BATCH_SIZE, PCA_COMPONENTS
)

from utils.utils import (
    fill_missing_values, write_results_to_file, selective_encode
)

# Создаем директории, если они не существуют
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Утилиты

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загрузка обучающего и тестового наборов данных.

    Args:
        train_path (str): Путь к обучающим данным.
        test_path (str): Путь к тестовым данным.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame для обучения и тестирования.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def data_preview(df: pd.DataFrame, dataset_name: str = 'Dataset') -> None:
    """
    Вывод и сохранение предварительного обзора DataFrame.

    Args:
        df (pd.DataFrame): DataFrame для обзора.
        dataset_name (str): Название набора данных для контекста.
    """
    preview = (
        f"--- {dataset_name} Head ---\n{df.head()}\n\n"
        f"--- {dataset_name} Info ---\n"
    )
    # Захват вывода df.info()
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    preview += buffer.getvalue() + "\n"

    preview += (
        f"--- {dataset_name} Description ---\n{df.describe()}\n\n"
        f"--- {dataset_name} Missing Values ---\n{df.isnull().sum()}\n\n"
    )
    print(preview)
    write_results_to_file(preview, filename=README_PATH)


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


def remove_outliers(df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
    """
    Удаляет выбросы из указанных колонок DataFrame на основе межквартильного размаха (IQR).

    Выбросы определяются как значения ниже Q1 - multiplier * IQR или выше Q3 + multiplier * IQR.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        columns (List[str]): Список колонок, из которых необходимо удалить выбросы.
        multiplier (float, optional): Множитель для расчета границ выбросов. По умолчанию 1.5.

    Returns:
        pd.DataFrame: DataFrame без выбросов в указанных колонках.
    """
    df_cleaned = df.copy()
    for column in columns:
        if column in df_cleaned.columns:
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            initial_shape = df_cleaned.shape
            df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
            final_shape = df_cleaned.shape
            removed = initial_shape[0] - final_shape[0]
            print(f"Удалено {removed} выбросов из колонки '{column}'.")
            write_results_to_file(f"Удалено {removed} выбросов из колонки '{column}'.", README_PATH)
        else:
            print(f"Колонка '{column}' не найдена в DataFrame.")
            write_results_to_file(f"Колонка '{column}' не найдена в DataFrame.", README_PATH)
    return df_cleaned


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


def label_encode(df: pd.DataFrame, columns: List[str]) -> Dict[str, LabelEncoder]:
    """
    Обучение и применение Label Encoding к указанным столбцам.

    Args:
        df (pd.DataFrame): DataFrame для кодирования.
        columns (List[str]): Список столбцов для Label Encoding.

    Returns:
        Dict[str, LabelEncoder]: Словарь с обученными LabelEncoders для каждого столбца.
    """
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return encoders


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Обучение модели Logistic Regression с использованием Grid Search для подбора оптимальных гиперпараметров.

    Args:
        X_train (np.ndarray): Обучающая матрица признаков.
        y_train (np.ndarray): Обучающие метки.
    Returns:
        LogisticRegression: Обученная модель Logistic Regression с лучшими найденными гиперпараметрами.
    """
    # Определение модели Logistic Regression
    logistic = LogisticRegression(solver='liblinear', random_state=42)

    # Определение сетки гиперпараметров для Grid Search
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Инверсная сила регуляризации
        'penalty': ['l1', 'l2']  # Тип регуляризации
    }

    # Инициализация GridSearchCV
    grid_search = GridSearchCV(
        estimator=logistic,
        param_grid=param_grid,
        scoring='f1',  # Метрика для оценки
        cv=5,  # Количество фолдов перекрестной проверки
        n_jobs=-1,  # Использование всех доступных процессоров
        verbose=1  # Уровень подробности вывода
    )

    # Обучение модели с Grid Search
    grid_search.fit(X_train, y_train)

    # Вывод лучших параметров и соответствующей метрики
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best Parameters for Logistic Regression: {best_params}")
    print(f"Best F1 Score from Grid Search: {best_score:.4f}")

    # Получение лучшей модели
    best_model = grid_search.best_estimator_

    return best_model


def random_forest_model_direct(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Обучение модели Random Forest.

    Args:
        X_train (np.ndarray): Обучающая матрица признаков.
        y_train (np.ndarray): Обучающие метки.

    Returns:
        RandomForestClassifier: Обученная модель Random Forest.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_val: np.ndarray, y_val: pd.Series, model_name: str) -> None:
    """
    Оценка модели и вывод метрик.

    Args:
        model: Обученная модель.
        X_val (np.ndarray): Валидационная матрица признаков.
        y_val (pd.Series): Валидационные метки.
        model_name (str): Название модели для вывода.
    """
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(classification_report(y_val, y_pred))
    write_results_to_file(f"--- {model_name} Evaluation ---", README_PATH)
    write_results_to_file(f"Accuracy: {acc:.4f}", README_PATH)
    write_results_to_file(f"F1 Score: {f1:.4f}", README_PATH)
    write_results_to_file(f"ROC AUC Score: {roc_auc:.4f}", README_PATH)
    report = classification_report(y_val, y_pred)
    write_results_to_file(report, README_PATH)

    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(PLOTS_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    write_results_to_file(f"ROC Curve saved as roc_curve_{model_name.lower().replace(' ', '_')}.png", README_PATH)


def train_neural_network(
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: optim.Optimizer,
        epochs: int,
        batch_size: int
) -> None:
    """
    Обучение нейронной сети.

    Args:
        model (nn.Module): Нейронная сеть.
        X_train (torch.Tensor): Обучающие данные.
        y_train (torch.Tensor): Обучающие метки.
        optimizer (optim.Optimizer): Оптимизатор.
        epochs (int): Количество эпох.
        batch_size (int): Размер батча.
    """
    criterion = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
            write_results_to_file(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}", README_PATH)


def evaluate_nn_model(y_pred: np.ndarray, y_true: pd.Series, model_name: str) -> None:
    """
    Оценка нейронной сети и вывод метрик.

    Args:
        y_pred (np.ndarray): Предсказанные метки.
        y_true (pd.Series): Истинные метки.
        model_name (str): Название модели для вывода.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(classification_report(y_true, y_pred))
    write_results_to_file(f"--- {model_name} Evaluation ---", README_PATH)
    write_results_to_file(f"Accuracy: {acc:.4f}", README_PATH)
    write_results_to_file(f"F1 Score: {f1:.4f}", README_PATH)
    write_results_to_file(f"ROC AUC Score: {roc_auc:.4f}", README_PATH)
    report = classification_report(y_true, y_pred)
    write_results_to_file(report, README_PATH)


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Выполнение корреляционного анализа и сохранение тепловой карты.

    Args:
        df (pd.DataFrame): DataFrame для анализа.
    """
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix.png'))
    plt.close()
    write_results_to_file("Correlation matrix saved as correlation_matrix.png", README_PATH)


def plot_boxplots(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Построение boxplot для числовых признаков и сохранение графика.

    Args:
        df (pd.DataFrame): DataFrame для построения графика.
    """
    plt.figure(figsize=(15, 10))
    df[columns].boxplot(rot=45)
    plt.title('Boxplots of Numerical Features')
    plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'))
    plt.close()
    write_results_to_file("Boxplots saved as boxplots.png", README_PATH)


def plot_scatter_with_target(df: pd.DataFrame, columns: List[str], target: str) -> None:
    """
    Построение scatter plot с целевой переменной и сохранение графика.

    Args:
        df (pd.DataFrame): DataFrame для построения графика.
        target (str): Название целевой переменной.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=target)
    plt.title(f'Scatter Plot of {columns[0]} vs {columns[1]}')
    plt.savefig(os.path.join(PLOTS_DIR, 'scatter_plot.png'))
    plt.close()
    write_results_to_file("Scatter plot saved as scatter_plot.png", README_PATH)


# --------------------
# Define MLP Models
# --------------------

class MLP1(nn.Module):
    """
    Нейронная сеть MLP1 с двумя скрытыми слоями.
    """

    def __init__(self, input_size: int):
        super(MLP1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, MLP1_HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(MLP1_HIDDEN_LAYERS[0], MLP1_HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(MLP1_HIDDEN_LAYERS[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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


def main(train_path: str, test_path: str) -> None:
    """
    Основная функция для выполнения всех этапов проекта:
    загрузка данных, предобработка, анализ, обучение моделей, оценка и сохранение результатов.

    Args:
        train_path (str): Путь к обучающему набору данных.
        test_path (str): Путь к тестовому набору данных.
    """
    # 1. Загрузка данных
    train_df, test_df = load_data(train_path, test_path)

    # 2. Предварительный просмотр данных
    data_preview(train_df, dataset_name='Training Data')
    data_preview(test_df, dataset_name='Testing Data')

    # 3. Предобработка данных: Заполнение пропусков
    train_df = fill_missing_values(train_df)
    test_df = fill_missing_values(test_df)

    # 4. Анализ данных
    correlation_analysis(train_df)  # Корреляционная матрица и тепловая карта
    plot_boxplots(train_df, columns=NUMERIC_COLUMNS)  # Boxplot для числовых признаков
    plot_scatter_with_target(train_df, target=TARGET_COLUMN,
                             columns=SCATTER_PLOT_COLUMNS)  # Scatter plot с целевой переменной


    # 5. Удаление выбросов из числовых колонок
    train_df = remove_outliers(train_df, columns=OUTLIER_COLUMNS, multiplier=1.5)

    # 6. Обучение LabelEncoders на обучающем наборе данных
    encoders = label_encode(train_df, LABEL_ENCODE_COLUMNS)

    # 7. Кодирование категориальных переменных
    train_df = selective_encode(train_df, ONE_HOT_COLUMNS, LABEL_ENCODE_COLUMNS, encoders)
    test_df = selective_encode(test_df, ONE_HOT_COLUMNS, LABEL_ENCODE_COLUMNS, encoders)

    # 8. Разделение данных на признаки и целевую переменную
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 9. Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_df_scaled = scaler.transform(test_df)

    # Сохранение scaler для инференса
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # 10. Применение PCA для снижения размерности
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    test_pca = pca.transform(test_df_scaled)

    # Сохранение PCA для инференса
    joblib.dump(pca, os.path.join(MODELS_DIR, 'pca.pkl'))

    # 11. Обучение моделей и их оценка

    # 11.1 Логистическая регрессия с Grid Search
    logistic_model = train_logistic_regression(X_train_pca, y_train)
    evaluate_model(logistic_model, X_val_pca, y_val, 'Logistic Regression')

    # 11.2 Случайный лес
    random_forest = random_forest_model_direct(X_train_pca, y_train)
    evaluate_model(random_forest, X_val_pca, y_val, 'Random Forest')

    # 11.3 Обучение и оценка MLP1
    mlp1 = MLP1(input_size=X_train_pca.shape[1])
    mlp1_optimizer = optim.Adam(mlp1.parameters(), lr=0.01)
    train_neural_network(
        model=mlp1,
        X_train=torch.FloatTensor(X_train_pca),
        y_train=torch.FloatTensor(y_train.values),
        optimizer=mlp1_optimizer,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE
    )
    mlp1.eval()
    with torch.no_grad():
        mlp1_outputs = mlp1(torch.FloatTensor(X_val_pca))
        mlp1_predictions = (mlp1_outputs.numpy().flatten() >= 0.5).astype(int)
    evaluate_nn_model(mlp1_predictions, y_val, 'MLP1')

    # 11.4 Обучение и оценка MLP2
    mlp2 = MLP2(input_size=X_train_pca.shape[1], hidden_layers=MLP2_HIDDEN_LAYERS)
    mlp2_optimizer = optim.Adam(mlp2.parameters(), lr=MLP_LEARNING_RATE)
    train_neural_network(
        model=mlp2,
        X_train=torch.FloatTensor(X_train_pca),
        y_train=torch.FloatTensor(y_train.values),
        optimizer=mlp2_optimizer,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE
    )
    mlp2.eval()
    with torch.no_grad():
        mlp2_outputs = mlp2(torch.FloatTensor(X_val_pca))
        mlp2_predictions = (mlp2_outputs.numpy().flatten() >= 0.5).astype(int)
    evaluate_nn_model(mlp2_predictions, y_val, 'MLP2')

    # 12. Сохранение предобработчиков и моделей

    # 12.1 Сохранение LabelEncoders
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))

    # 12.2 Сохранение модели MLP2
    torch.save(mlp2.state_dict(), os.path.join(MODELS_DIR, 'mlp2.pth'))

    # Сообщение об успешном завершении
    success_message = "Обучение моделей завершено и объекты сохранены."
    print(success_message)
    write_results_to_file(success_message, README_PATH)


if __name__ == '__main__':
    main(TRAIN_PATH, TEST_PATH)
