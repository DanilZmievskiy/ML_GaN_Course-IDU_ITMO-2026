feature_names = training_points.columns
importances = np.abs(w)

feature_importance = pd.DataFrame({
    'Название признака': feature_names,
    'Важность признака': importances
})

feature_importance = feature_importance.sort_values('Важность признака', ascending=False)

print("\n=== Важность признаков ===")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Название признака'], feature_importance['Важность признака'])
plt.xlabel('Важность (модуль веса)')
plt.title('Важность признаков в модели')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()