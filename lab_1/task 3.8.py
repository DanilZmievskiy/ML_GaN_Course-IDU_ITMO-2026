X_all = np.vstack([training_points, test_points])
X_all_scaled = scaler.transform(X_all)
y_all_pred_scaled = X_all_scaled @ w + b
y_all_pred = y_all_pred_scaled * y_std + y_mean

y_all_true = np.concatenate([training_values, test_values])

errors = y_all_true - y_all_pred

threshold = -2 * errors.std()
anomalies_mask = errors < threshold
anomalies = pd.DataFrame({
    'Реальная цена': y_all_true[anomalies_mask],
    'Предсказанная цена': y_all_pred[anomalies_mask],
    'Разница': errors[anomalies_mask]
})

print(f"\n=== Аномально низкие цены (ошибка < {threshold:.2f}) ===")
print(f"Найдено {len(anomalies)} объектов с аномально заниженной ценой")
print("\nПервые 10 аномалий:")
print(anomalies.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
plt.scatter(y_all_true, y_all_pred, alpha=0.3, label='Все объекты')
plt.scatter(anomalies['Реальная цена'], anomalies['Предсказанная цена'], 
            color='red', s=50, label='Аномально низкие цены')
plt.plot([y_all_true.min(), y_all_true.max()], 
         [y_all_true.min(), y_all_true.max()], 'k--', label='Идеальное предсказание')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Аномально низкие цены (ошибка > 2σ)')
plt.legend()
plt.show()