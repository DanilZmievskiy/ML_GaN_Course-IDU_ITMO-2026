import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_data(n_points=20):
    X = np.linspace(-5, 5, n_points)
    y = 10 * X - 7

    X_train = X[0::2].reshape(-1, 1)
    y_train = y[0::2] + np.random.randn(int(n_points/2)) * 10

    X_test = X[1::2].reshape(-1, 1)
    y_test = y[1::2] + np.random.randn(int(n_points/2)) * 10

    print(f'Generated {len(X_train)} train samples and {len(X_test)} test samples')
    return X, X_train, y_train, X_test, y_test

X, X_train, y_train, X_test, y_test = generate_data(100)

X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

w = np.linalg.pinv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

b = w[0]
coef = w[1:]

print(f"\nРучная реализация МНК:")
print(f"b = {b:.4f}")
print(f"w = {coef[0]:.4f}")
print(f"Уравнение: y = {coef[0]:.4f} * x + {b:.4f}")

y_train_pred_manual = X_train_aug @ w
X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_test_pred_manual = X_test_aug @ w

model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

y_train_pred_sklearn = model_sklearn.predict(X_train)
y_test_pred_sklearn = model_sklearn.predict(X_test)

print(f"\nsklearn LinearRegression:")
print(f"b = {model_sklearn.intercept_:.4f}")
print(f"w = {model_sklearn.coef_[0]:.4f}")

print("\n=== Метрики качества ===")
print(f"{'Метрика':<15} {'Ручная реализация (train)':<25} {'Ручная реализация (test)':<25} {'sklearn (train)':<20} {'sklearn (test)':<20}")
print("-" * 105)
print(f"{'MSE':<15} {mean_squared_error(y_train, y_train_pred_manual):<25.4f} {mean_squared_error(y_test, y_test_pred_manual):<25.4f} {mean_squared_error(y_train, y_train_pred_sklearn):<20.4f} {mean_squared_error(y_test, y_test_pred_sklearn):<20.4f}")
print(f"{'MAE':<15} {mean_absolute_error(y_train, y_train_pred_manual):<25.4f} {mean_absolute_error(y_test, y_test_pred_manual):<25.4f} {mean_absolute_error(y_train, y_train_pred_sklearn):<20.4f} {mean_absolute_error(y_test, y_test_pred_sklearn):<20.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, model_sklearn.predict(X.reshape(-1, 1)), label='predicted', color='red')
plt.legend(loc='best')
plt.title('Линейная регрессия')
plt.show()