import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

url = "https://www.kaggle.com/datasets/vikrishnan/boston-house-prices/"
data = pd.read_csv(url)

print("Первые 5 строк:")
print(data.head())
print("\nПризнаки:", list(data.columns[:-1]))
print("Цена (то что предсказываем): medv")

X = data.drop('medv', axis=1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n=== Ошибки модели ===")
print(f"Обучающая выборка - MSE: {mean_squared_error(y_train, y_pred_train):.2f}")
print(f"Обучающая выборка - MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Тестовая выборка   - MSE: {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"Тестовая выборка   - MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Реальные vs предсказанные цены (тестовая выборка)')
plt.show()