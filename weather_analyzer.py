
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = pd.read_csv("weather_data.csv")

# Clean and preprocess data
data = data.dropna(subset=['Temperature', 'Humidity', 'Rainfall', 'Year'])

# Fill missing values if any
data['Humidity'].fillna(data['Humidity'].mean(), inplace=True)
data['Rainfall'].fillna(data['Rainfall'].mean(), inplace=True)

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Line Chart: Temperature over Years
plt.figure(figsize=(8,4))
sns.lineplot(x='Year', y='Temperature', data=data, marker='o')
plt.title("Temperature Trends Over Years")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Graph: Rainfall over Years
plt.figure(figsize=(8,4))
sns.barplot(x='Year', y='Rainfall', data=data, color='royalblue')
plt.title("Yearly Rainfall Distribution")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot: Temperature vs Humidity
plt.figure(figsize=(6,4))
sns.scatterplot(x='Temperature', y='Humidity', data=data, color='green')
plt.title("Humidity vs Temperature Correlation")
plt.tight_layout()
plt.show()

# Linear Regression to predict future temperature
X = data[['Year']]
y = data['Temperature']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & trendline
future_years = pd.DataFrame({'Year': list(range(2025, 2031))})
future_temp = model.predict(future_years)

plt.figure(figsize=(8,4))
sns.scatterplot(x='Year', y='Temperature', data=data, color='blue', label='Actual')
plt.plot(future_years['Year'], future_temp, color='orange', label='Prediction', marker='o')
plt.title("Temperature Prediction for Next Years")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Error metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== Statistical Summary ===")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
