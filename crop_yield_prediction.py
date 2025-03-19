# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("pesticides.csv")  

df = df.dropna() 

X = df[['Year']]
y = df['Value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2)  
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared Score (R2): {r2}")

plt.figure(figsize=(10, 6))  


sns.scatterplot(x=X_test[:, 1], y=y_test, color='blue', label='Actual', s=50)


sns.scatterplot(x=X_test[:, 1], y=y_pred, color='red', label='Predicted', s=50, alpha=0.8)


plt.xlabel("Year", fontsize=12)
plt.ylabel("Crop Yield (Pesticide Usage)", fontsize=12)
plt.title("Actual vs Predicted Crop Yield", fontsize=14)
plt.legend()
plt.show()
