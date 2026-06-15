mae_manual = mean_absolute_error(test_values, y_pred_test_manual)
mse_manual = mean_squared_error(test_values, y_pred_test_manual)
rmse_manual = np.sqrt(mse_manual)

print("\n=== Сравнение моделей ===")
print(f"{'Метрика':<10} {'sklearn':<15} {'Моя реализация':<15}")
print("-" * 40)
print(f"{'MAE':<10} {mean_absolute_error_linear_model:<15.2f} {mae_manual:<15.2f}")
print(f"{'MSE':<10} {mean_squared_error_linear_model:<15.2f} {mse_manual:<15.2f}")
print(f"{'RMSE':<10} {np.sqrt(mean_squared_error_linear_model):<15.2f} {rmse_manual:<15.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(test_values, test_predictions_linear, alpha=0.5)
plt.plot([test_values.min(), test_values.max()], [test_values.min(), test_values.max()], 'r--')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('sklearn LinearRegression')

plt.subplot(1, 2, 2)
plt.scatter(test_values, y_pred_test_manual, alpha=0.5, color='green')
plt.plot([test_values.min(), test_values.max()], [test_values.min(), test_values.max()], 'r--')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Моя реализация (SGD + регуляризация)')

plt.tight_layout()
plt.show()