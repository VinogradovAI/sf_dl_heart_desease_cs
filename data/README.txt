Ошибка при загрузке предобработчиков: [Errno 2] No such file or directory: 'models\\scaler.pkl'
--- Training Data Head ---
   ID        age  sex     chest  ...  slope  number_of_major_vessels  thal  class
0   0  49.207124    0  4.000000  ...      2                        0     3      1
1   1  53.628425    1  1.741596  ...      2                        0     3      0
2   2  49.591426    1  4.000000  ...      2                        2     7      1
3   3  58.991445    1  4.000000  ...      1                        1     7      1
4   4  51.053602    1  1.954609  ...      1                        1     3      0

[5 rows x 15 columns]

--- Training Data Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 600000 entries, 0 to 599999
Data columns (total 15 columns):
 #   Column                                Non-Null Count   Dtype  
---  ------                                --------------   -----  
 0   ID                                    600000 non-null  int64  
 1   age                                   600000 non-null  float64
 2   sex                                   600000 non-null  int64  
 3   chest                                 600000 non-null  float64
 4   resting_blood_pressure                600000 non-null  float64
 5   serum_cholestoral                     600000 non-null  float64
 6   fasting_blood_sugar                   600000 non-null  int64  
 7   resting_electrocardiographic_results  600000 non-null  int64  
 8   maximum_heart_rate_achieved           600000 non-null  float64
 9   exercise_induced_angina               600000 non-null  int64  
 10  oldpeak                               600000 non-null  float64
 11  slope                                 600000 non-null  int64  
 12  number_of_major_vessels               600000 non-null  int64  
 13  thal                                  600000 non-null  int64  
 14  class                                 600000 non-null  int64  
dtypes: float64(6), int64(9)
memory usage: 68.7 MB

--- Training Data Description ---
                  ID            age  ...           thal          class
count  600000.000000  600000.000000  ...  600000.000000  600000.000000
mean   299999.500000      54.426085  ...       4.711378       0.444185
std    173205.225094       9.086041  ...       1.934766       0.496875
min         0.000000      26.061695  ...       3.000000       0.000000
25%    149999.750000      48.078493  ...       3.000000       0.000000
50%    299999.500000      55.133425  ...       3.000000       0.000000
75%    449999.250000      60.663775  ...       7.000000       1.000000
max    599999.000000      79.591647  ...       7.000000       1.000000

[8 rows x 15 columns]

--- Training Data Missing Values ---
ID                                      0
age                                     0
sex                                     0
chest                                   0
resting_blood_pressure                  0
serum_cholestoral                       0
fasting_blood_sugar                     0
resting_electrocardiographic_results    0
maximum_heart_rate_achieved             0
exercise_induced_angina                 0
oldpeak                                 0
slope                                   0
number_of_major_vessels                 0
thal                                    0
class                                   0
dtype: int64


--- Testing Data Head ---
       ID        age  sex  ...  slope  number_of_major_vessels  thal
0  600000  53.963191    1  ...      1                        2     7
1  600001  49.621479    1  ...      1                        1     7
2  600002  36.933893    1  ...      1                        0     7
3  600003  54.884588    1  ...      1                        1     7
4  600004  71.016392    0  ...      1                        1     3

[5 rows x 14 columns]

--- Testing Data Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400000 entries, 0 to 399999
Data columns (total 14 columns):
 #   Column                                Non-Null Count   Dtype  
---  ------                                --------------   -----  
 0   ID                                    400000 non-null  int64  
 1   age                                   400000 non-null  float64
 2   sex                                   400000 non-null  int64  
 3   chest                                 400000 non-null  float64
 4   resting_blood_pressure                400000 non-null  float64
 5   serum_cholestoral                     400000 non-null  float64
 6   fasting_blood_sugar                   400000 non-null  int64  
 7   resting_electrocardiographic_results  400000 non-null  int64  
 8   maximum_heart_rate_achieved           400000 non-null  float64
 9   exercise_induced_angina               400000 non-null  int64  
 10  oldpeak                               400000 non-null  float64
 11  slope                                 400000 non-null  int64  
 12  number_of_major_vessels               400000 non-null  int64  
 13  thal                                  400000 non-null  int64  
dtypes: float64(6), int64(8)
memory usage: 42.7 MB

--- Testing Data Description ---
                  ID            age  ...  number_of_major_vessels          thal
count  400000.000000  400000.000000  ...            400000.000000  400000.00000
mean   799999.500000      54.406917  ...                 0.681432       4.70715
std    115470.198175       9.101044  ...                 0.950323       1.93353
min    600000.000000      27.496874  ...                 0.000000       3.00000
25%    699999.750000      48.021226  ...                 0.000000       3.00000
50%    799999.500000      55.130138  ...                 0.000000       3.00000
75%    899999.250000      60.667833  ...                 1.000000       7.00000
max    999999.000000      80.751628  ...                 3.000000       7.00000

[8 rows x 14 columns]

--- Testing Data Missing Values ---
ID                                      0
age                                     0
sex                                     0
chest                                   0
resting_blood_pressure                  0
serum_cholestoral                       0
fasting_blood_sugar                     0
resting_electrocardiographic_results    0
maximum_heart_rate_achieved             0
exercise_induced_angina                 0
oldpeak                                 0
slope                                   0
number_of_major_vessels                 0
thal                                    0
dtype: int64


Correlation matrix saved as correlation_matrix.png
Boxplots saved as boxplots.png
Scatter plot saved as scatter_plot.png
Удалено 21129 выбросов из колонки 'resting_blood_pressure'.
Удалено 23496 выбросов из колонки 'serum_cholestoral'.
Удалено 83657 выбросов из колонки 'fasting_blood_sugar'.
Удалено 2665 выбросов из колонки 'maximum_heart_rate_achieved'.
Удалено 7608 выбросов из колонки 'oldpeak'.
--- Logistic Regression Evaluation ---
Accuracy: 0.8338
F1 Score: 0.8044
ROC AUC Score: 0.9110
              precision    recall  f1-score   support

           0       0.85      0.87      0.86     52429
           1       0.82      0.79      0.80     39860

    accuracy                           0.83     92289
   macro avg       0.83      0.83      0.83     92289
weighted avg       0.83      0.83      0.83     92289

ROC Curve saved as roc_curve_logistic_regression.png
--- Random Forest Evaluation ---
Accuracy: 0.8405
F1 Score: 0.8130
ROC AUC Score: 0.9146
              precision    recall  f1-score   support

           0       0.85      0.87      0.86     52429
           1       0.82      0.80      0.81     39860

    accuracy                           0.84     92289
   macro avg       0.84      0.84      0.84     92289
weighted avg       0.84      0.84      0.84     92289

ROC Curve saved as roc_curve_random_forest.png
Epoch [1/50], Loss: 0.3688
Epoch [10/50], Loss: 0.3609
Epoch [20/50], Loss: 0.3612
Epoch [30/50], Loss: 0.3611
Epoch [40/50], Loss: 0.3612
Epoch [50/50], Loss: 0.3610
--- MLP1 Evaluation ---
Accuracy: 0.8410
F1 Score: 0.8150
ROC AUC Score: 0.8374
              precision    recall  f1-score   support

           0       0.86      0.86      0.86     52429
           1       0.82      0.81      0.81     39860

    accuracy                           0.84     92289
   macro avg       0.84      0.84      0.84     92289
weighted avg       0.84      0.84      0.84     92289

Epoch [1/50], Loss: 0.3678
Epoch [10/50], Loss: 0.3524
Epoch [20/50], Loss: 0.3496
Epoch [30/50], Loss: 0.3480
Epoch [40/50], Loss: 0.3476
Epoch [50/50], Loss: 0.3465
--- MLP2 Evaluation ---
Accuracy: 0.8473
F1 Score: 0.8226
ROC AUC Score: 0.8440
              precision    recall  f1-score   support

           0       0.86      0.87      0.87     52429
           1       0.83      0.82      0.82     39860

    accuracy                           0.85     92289
   macro avg       0.84      0.84      0.84     92289
weighted avg       0.85      0.85      0.85     92289

Обучение моделей завершено и объекты сохранены.
Предобработчики успешно загружены.
Модель успешно загружена.
Тестовые данные успешно загружены.
Предобработка тестовых данных завершена.
Данные конвертированы в тензор.
Инференс выполнен успешно.
Предсказания сохранены в data/submission.csv
