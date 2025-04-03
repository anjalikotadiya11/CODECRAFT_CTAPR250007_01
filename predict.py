import joblib  
import numpy as np  

# Load the saved model
model = joblib.load("house_price_model.pkl")

# Example new house details: [Square Footage, Bedrooms, Bathrooms]
new_house = np.array([[2000, 3, 2]])  # Change values as needed

# Predict the price
predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
