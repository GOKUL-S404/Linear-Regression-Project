import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Area": [
        600, 800, 1000, 1200, 1500, 1800, 2000, 2200,
        700, 900, 1100, 1300, 1600, 1900, 2100, 2400,
        750, 950, 1150, 1400, 1700, 1950, 2300, 2500,
        650, 850, 1050, 1250, 1550, 1850, 2150, 2450,
        720, 920, 1180, 1450, 1750, 2050, 2350, 2600
    ],
    "Bedrooms": [
        1, 2, 2, 3, 3, 3, 4, 4,
        1, 2, 2, 3, 3, 4, 4, 4,
        1, 2, 3, 3, 3, 4, 4, 5,
        1, 2, 2, 3, 3, 4, 4, 5,
        1, 2, 3, 3, 4, 4, 5, 5
    ],
    "Age": [
        15, 10, 8, 5, 4, 3, 2, 1,
        12, 9, 7, 6, 4, 3, 2, 1,
        14, 11, 7, 6, 5, 3, 2, 1,
        16, 12, 9, 6, 5, 4, 2, 1,
        13, 10, 7, 5, 4, 3, 2, 1
    ],
    "Distance": [
        12, 10, 9, 8, 7, 6, 5, 4,
        13, 11, 9, 8, 7, 5, 4, 3,
        14, 11, 9, 7, 6, 5, 4, 3,
        15, 12, 10, 8, 7, 6, 4, 3,
        13, 10, 8, 7, 6, 5, 3, 2
    ],
    "Price": [
        20, 28, 35, 45, 55, 65, 75, 85,
        22, 30, 38, 48, 58, 72, 80, 95,
        23, 32, 43, 52, 62, 76, 92, 105,
        19, 29, 36, 47, 57, 69, 88, 102,
        21, 33, 44, 54, 68, 79, 98, 112
    ]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

X = df[["Area", "Bedrooms", "Age", "Distance"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

print("\n--- Model Coefficients ---")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(2)
})

print("\n--- Actual vs Predicted ---")
print(results.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- New House Price Prediction ---")

area = float(input("Enter house area in sq ft: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = float(input("Enter house age in years: "))
distance = float(input("Enter distance from city centre in km: "))

new_house = pd.DataFrame(
    [[area, bedrooms, age, distance]],
    columns=["Area", "Bedrooms", "Age", "Distance"]
)

predicted_price = model.predict(new_house)[0]

print(f"\nEstimated House Price: Rs. {predicted_price:.2f} Lakhs")
