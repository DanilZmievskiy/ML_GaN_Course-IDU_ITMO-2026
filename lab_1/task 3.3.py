import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

degrees = [1, 3, 5, 10]

print("Результаты:\n")
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    
    mse = np.mean((y - y_pred)**2)
    print(f"Степень {d}: MSE = {mse:.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.plot(X, np.sin(X).ravel(), 'b-', label='Истинная функция sin(x)')
    plt.plot(X, y_pred, 'r--', label=f'Полином степени {d}')
    plt.scatter(X, y, s=10, c='gray', alpha=0.5, label='Данные с шумом')
    plt.legend()
    plt.title(f'Степень {d} (MSE = {mse:.4f})')
    plt.show()
print("\n=== ВЫВОД ===")
print("Лучше всего подходит степень 5.")
print("При степени 1 и 3: модель слишком простая, не может описать изгибы синусоиды (недообучение).")
print("При степени 10: модель начинает колебаться и запоминать шум (переобучение).")
print("Степень 5 — золотая середина.")