import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_wave_set(n_support=1000, n_train=250, std=0.3):
    data = {}
    data['support'] = np.linspace(0, 2*np.pi, num=n_support)
    data['values'] = np.sin(data['support']) + 1
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']) + 1 + np.random.normal(0, std, size=data['x_train'].shape[0])
    return data

data = generate_wave_set(1000, 250)

X_train = data['x_train'].reshape(-1, 1)
y_train = data['y_train']
X_support = data['support'].reshape(-1, 1)
y_true = data['values']

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_support)

mse_linear = mean_squared_error(y_true, y_pred_linear)
mae_linear = mean_absolute_error(y_true, y_pred_linear)

print("=== Обычная линейная регрессия ===")
print(f"Уравнение: y = {model_linear.coef_[0]:.4f} * x + {model_linear.intercept_:.4f}")
print(f"MSE на тесте: {mse_linear:.4f}")
print(f"MAE на тесте: {mae_linear:.4f}")

degrees = [3, 5, 10]

print("\n=== Полиномиальная регрессия ===")
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_support_poly = poly.transform(X_support)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_support_poly)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\nСтепень полинома = {deg}")
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='Истинная функция')
plt.scatter(data['x_train'], data['y_train'], 20, 'g', alpha=0.6, label='Обучающие точки')
plt.legend()
plt.title('Исходные данные')

plt.subplot(1, 3, 2)
plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='Истинная функция')
plt.plot(data['support'], y_pred_linear, 'r-', label='Линейная регрессия')
plt.scatter(data['x_train'], data['y_train'], 20, 'g', alpha=0.6)
plt.legend()
plt.title('Обычная линейная регрессия (плохо)')

plt.subplot(1, 3, 3)
poly5 = PolynomialFeatures(degree=5)
X_train_poly5 = poly5.fit_transform(X_train)
X_support_poly5 = poly5.transform(X_support)
model5 = LinearRegression()
model5.fit(X_train_poly5, y_train)
y_pred5 = model5.predict(X_support_poly5)

plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='Истинная функция')
plt.plot(data['support'], y_pred5, 'r-', label='Полином степени 5')
plt.scatter(data['x_train'], data['y_train'], 20, 'g', alpha=0.6)
plt.legend()
plt.title('Полиномиальная регрессия (хорошо)')

plt.tight_layout()
plt.show()

print("\n=== ВЫВОДЫ ===")
print("1. Обычная линейная регрессия не может описать синусоиду — ошибки большие.")
print("2. Полиномиальная регрессия подходит гораздо лучше.")
print("3. Слишком высокая степень (10) может привести к переобучению.")