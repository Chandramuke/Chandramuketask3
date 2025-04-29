import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('C:/Users/91636/OneDrive/Desktop/Housing.csv')
data.head()

print(data.isnull().sum())

df_simple = data.copy()
X_simple = df_simple[['area']]
y_simple = df_simple['price']


features_multi = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X_multi = data[features_multi]
y_multi = data['price']

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)


X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)
     

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)


multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)
y_pred_simple = simple_model.predict(X_test_simple)
mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("\nSimple Linear Regression Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_simple:.2f}")
print(f"Mean Squared Error (MSE): {mse_simple:.2f}")
print(f"R-squared (R²): {r2_simple:.2f}")


y_pred_multi = multi_model.predict(X_test_multi)
mae_multi = mean_absolute_error(y_test_multi, y_pred_multi)
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("\nMultiple Linear Regression Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_multi:.2f}")
print(f"Mean Squared Error (MSE): {mse_multi:.2f}")
print(f"R-squared (R²): {r2_multi:.2f}")
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


plt.scatter(y_test_multi, y_pred_multi, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Multiple Linear Regression)")
plt.show()plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


plt.scatter(y_test_multi, y_pred_multi, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Multiple Linear Regression)")
plt.show()
print("\nSimple Linear Regression Coefficients:")
print(f"Intercept: {simple_model.intercept_}")
print(f"Coefficient for Area: {simple_model.coef_[0]}")


print("\nMultiple Linear Regression Coefficients:")
print(f"Intercept: {multi_model.intercept_}")
coefficients_df = pd.DataFrame({
    'Feature': features_multi,
    'Coefficient': multi_model.coef_
})
print(coefficients_df)
print("\nSummary:")
summary_table = pd.DataFrame({
    'Model': ['Simple Linear Regression', 'Multiple Linear Regression'],
    'MAE': [mae_simple, mae_multi],
    'MSE': [mse_simple, mse_multi],
    'R-squared': [r2_simple, r2_multi]
})
print(summary_table)