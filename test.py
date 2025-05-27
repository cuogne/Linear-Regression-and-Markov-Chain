import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/customer_purchase_behaviors.csv')

data['age_sqrt'] = data['age'] ** 0.5
data['annual_income_sqrt'] = data['annual_income'] ** 0.5
data['purchase_amount_sqrt'] = data['purchase_amount'] ** 0.5
data['purchase_frequency_sqrt'] = data['purchase_frequency'] ** 0.5

features = ['age_sqrt', 'annual_income_sqrt', 'purchase_amount_sqrt', 'purchase_frequency_sqrt']
X = data[features]
y = data['loyalty_score']

train_size = int(0.8 * len(X))

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Tính MSE trên tập test
mse = mean_squared_error(y_test, y_pred)

# In công thức mô hình
coefficients = model.coef_
intercept = model.intercept_
equation = f"Y = {intercept:.6f}"
for coef, name in zip(coefficients, features):
    equation += f" + ({coef:.6f}) * {name}"
print(f"Mô hình: {equation}")
print(f"MSE trên tập test: {mse:.6f}")

# In các hệ số để đánh giá tầm quan trọng
coef_data = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
print("\nHệ số của các đặc trưng:")
print(coef_data.sort_values(by='Coefficient', key=abs, ascending=False))

import matplotlib.pyplot as plt

# Vẽ đồ thị phân tán giữa giá trị thực và giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực (y_test)')
plt.ylabel('Giá trị dự đoán (y_pred)')
plt.title('Đồ thị phân tán: Giá trị thực vs Giá trị dự đoán')
plt.grid(True, alpha=0.3)
plt.show()