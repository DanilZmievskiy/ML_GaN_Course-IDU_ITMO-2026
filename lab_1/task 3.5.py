import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(training_points)
X_test_scaled = scaler.transform(test_points)

y_mean = training_values.mean()
y_std = training_values.std()
if y_std == 0:
    y_std = 1
y_train_scaled = (training_values - y_mean) / y_std

learning_rate = 0.01
n_iterations = 500
batch_size = 64
lambda_l2 = 0.01
lambda_l1 = 0.001
tol = 1e-6

n_features = X_train_scaled.shape[1]
w = np.zeros(n_features)
b = 0.0

loss_history = []
prev_loss = float('inf')

np.random.seed(42)

print("Начинаем обучение...")

for epoch in range(n_iterations):
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train_scaled[indices]
    
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        y_pred_batch = X_batch @ w + b
        errors = y_pred_batch - y_batch
        
        grad_w = (X_batch.T @ errors) / len(y_batch)
        grad_b = errors.mean()
        
        grad_w += lambda_l2 * w
        grad_w += lambda_l1 * np.sign(w)
        
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    y_pred_train = X_train_scaled @ w + b
    mse = ((y_pred_train - y_train_scaled) ** 2).mean()
    reg = (lambda_l2 / 2) * (w ** 2).sum() + lambda_l1 * np.abs(w).sum()
    loss = mse + reg
    loss_history.append(loss)
    
    if epoch > 0 and abs(prev_loss - loss) < tol:
        print(f"Остановка на эпохе {epoch}, изменение loss < {tol}")
        break
    prev_loss = loss
    
    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch+1}/{n_iterations}, Loss = {loss:.6f}")

print("\nОбучение завершено!")

y_pred_test_scaled = X_test_scaled @ w + b
y_pred_test_manual = y_pred_test_scaled * y_std + y_mean