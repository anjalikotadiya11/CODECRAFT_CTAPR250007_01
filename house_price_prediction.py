import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
import joblib  

# Load the dataset
df = pd.read_csv("train.csv")

# Select relevant columns
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]  # Square Footage, Bedrooms, Bathrooms
target = "SalePrice"  # House Price

# Create input (X) and output (y) variables
X = df[features]
y = df[target]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show dataset sizes
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Save the trained model
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")
