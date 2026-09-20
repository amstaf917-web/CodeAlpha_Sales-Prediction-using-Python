import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

data = pd.read_csv("Advertising.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())


# --------------------------------------------------
# 2. Clean the data
# --------------------------------------------------

# Remove unnecessary index column if it exists
if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with missing values
data = data.dropna()

print("\nDataset after cleaning:")
print(data.shape)


# --------------------------------------------------
# 3. Basic statistical analysis
# --------------------------------------------------

print("\nStatistical Summary:")
print(data.describe())


# --------------------------------------------------
# 4. Select features and target
# --------------------------------------------------

# Advertising platforms are used as input features
X = data[["TV", "Radio", "Newspaper"]]

# Sales is the value we want to predict
y = data["Sales"]


# --------------------------------------------------
# 5. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 6. Create the regression model
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))


# --------------------------------------------------
# 10. Display model coefficients
# --------------------------------------------------

print("\nAdvertising Impact")
print("-------------------------")
print("TV coefficient:", round(model.coef_[0], 3))
print("Radio coefficient:", round(model.coef_[1], 3))
print("Newspaper coefficient:", round(model.coef_[2], 3))
print("Intercept:", round(model.intercept_, 3))


# --------------------------------------------------
# 11. Compare actual and predicted sales
# --------------------------------------------------

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head(10))


# --------------------------------------------------
# 12. Analyze advertising impact
# --------------------------------------------------

impact = pd.DataFrame({
    "Advertising Platform": ["TV", "Radio", "Newspaper"],
    "Coefficient": model.coef_
})

print("\nAdvertising Impact:")
print(impact)


# --------------------------------------------------
# 13. Visualize actual vs predicted sales
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14. Visualize advertising spend vs sales
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(data["TV"], data["Sales"], alpha=0.6)

plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("TV Advertising Spend vs Sales")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(data["Radio"], data["Sales"], alpha=0.6)

plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.title("Radio Advertising Spend vs Sales")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(data["Newspaper"], data["Sales"], alpha=0.6)

plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.title("Newspaper Advertising Spend vs Sales")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 15. Predict sales for a new advertising budget
# --------------------------------------------------

new_advertising = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

predicted_sales = model.predict(new_advertising)

print("\nPredicted Sales for New Advertising Budget:")
print(round(predicted_sales[0], 2))