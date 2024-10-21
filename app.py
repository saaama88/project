from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model (Ensure you have a model saved as flipkart_model.pkl)
model = joblib.load('model.pkl')

# Assuming the same scaler used during training
scaler = joblib.load('scaler .pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    # Extracting input data from form
    features = {
        'Brand': [request.form['brand']], 
        'Model': [request.form['model']], 
        'Color': [request.form['color']],
        'RAM': [float(request.form['ram'])], 
        'Storage': [request.form['storage']], 
        'Rating': [float(request.form['rating'])],
        'Original Price': [float(request.form['original_price'])]
    }

    # Create a DataFrame from the input
    data = pd.DataFrame(features)

    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=['Brand', 'Model', 'Color', 'RAM', 'Storage'], drop_first=True, dtype=int)

    # Normalize numerical features
    data_encoded[['Original Price', 'Rating']] = scaler.transform(data_encoded[['Original Price', 'Rating']])

    # Make sure the input data has the same columns as the training data
    # In practice, you might need to align the columns with the training set
    missing_cols = set(model.feature_names_in_) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0

    data_encoded = data_encoded[model.feature_names_in_]

    # Make a prediction
    prediction = model.predict(data_encoded)

    return render_template('result.html', prediction_text=f'Predicted Price: {prediction}')


if __name__ == "__main__":
    app.run(debug=True)
