from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = "Models/best_random_forest_model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Convert JSON data to DataFrame
    input_data = pd.DataFrame(data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
