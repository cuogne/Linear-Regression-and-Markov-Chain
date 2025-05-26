import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
data = pd.read_csv('data/customer_purchase_behaviors.csv')

# Tạo các đặc trưng mới
data['age_squared'] = data['age'] ** 2
data['annual_income_squared'] = data['annual_income'] ** 2
data['purchase_amount_squared'] = data['purchase_amount'] ** 2
data['purchase_frequency_squared'] = data['purchase_frequency'] ** 2
data['purchase_amount_freq_interaction'] = data['purchase_amount'] * data['purchase_frequency']
data['age_purchase_interaction'] = data['age'] * data['purchase_amount']
data['sqrt_annual_income'] = np.sqrt(data['annual_income'])
data['sqrt_purchase_amount'] = np.sqrt(data['purchase_amount'])

# Lựa chọn các đặc trưng cho mô hình
features = [
    'age', 'age_squared',
    'annual_income', 'sqrt_annual_income', 'annual_income_squared',
    'purchase_amount', 'sqrt_purchase_amount', 'purchase_amount_squared',
    'purchase_frequency', 'purchase_frequency_squared',
    'purchase_amount_freq_interaction', 'age_purchase_interaction'
]
X = data[features]
y = data['loyalty_score']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_scaled, y)

# Dự đoán và tính MSE
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)

# In công thức mô hình
coefficients = model.coef_
intercept = model.intercept_
feature_names = X.columns
equation = f"Y = {intercept:.6f}"
for coef, name in zip(coefficients, feature_names):
    equation += f" + ({coef:.6f}) * {name}"
print(f"Mô hình: {equation}")
print(f"MSE: {mse:.6f}")

# In các hệ số để đánh giá tầm quan trọng
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print("\nHệ số của các đặc trưng:")
print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False))

# Mô hình: Y = 6.794118 + (-2.623579) * age + (6.999702) * age_squared + (38.657126) * annual_income + (-22.850043) * sqrt_annual_income + (-15.310514) * annual_income_squared + (15.456039) * purchase_amount + (-7.309940) * sqrt_purchase_amount + (6.631924) * purchase_amount_squared + (-2.863824) * purchase_frequency + (9.045988) * purchase_frequency_squared + (-14.319396) * purchase_amount_freq_interaction + (-9.731166) * age_purchase_interaction
# MSE: 0.015307

# Hệ số của các đặc trưng:
#                              Feature  Coefficient
# 2                      annual_income    38.657126
# 3                 sqrt_annual_income   -22.850043
# 5                    purchase_amount    15.456039
# 4              annual_income_squared   -15.310514
# 10  purchase_amount_freq_interaction   -14.319396
# 11          age_purchase_interaction    -9.731166
# 9         purchase_frequency_squared     9.045988
# 6               sqrt_purchase_amount    -7.309940
# 1                        age_squared     6.999702
# 7            purchase_amount_squared     6.631924
# 8                 purchase_frequency    -2.863824
# 0                                age    -2.623579