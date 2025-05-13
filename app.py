from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the dataset
DATA_FILE = 'bengaluru_borewell_dataset.csv'
data = pd.read_csv(DATA_FILE)

# Ensure the dataset contains the required columns
REQUIRED_COLUMNS = ['latitude', 'longitude', 'feasibility']
if not all(col in data.columns for col in REQUIRED_COLUMNS):
    raise ValueError(f"The dataset must contain the following columns: {REQUIRED_COLUMNS}")

@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        # Parse input data
        data_input = request.get_json()
        latitude = data_input.get('latitude')
        longitude = data_input.get('longitude')

        # Validate input
        if latitude is None or longitude is None:
            return jsonify({
                "feasibility": 0,
                "confidence": 0,
                "reason": "Invalid input: Latitude and Longitude are required."
            }), 400

        # Calculate distances to find the closest match in the dataset
        data['distance'] = np.sqrt((data['latitude'] - latitude)**2 + (data['longitude'] - longitude)**2)
        closest_row = data.loc[data['distance'].idxmin()]

        # Extract feasibility and other details
        feasibility = int(closest_row['feasibility'])
        reason = "Closest match found in the dataset."
        confidence = 100  # Confidence is 100% since it's directly from the dataset

        # Return the result
        return jsonify({
            "feasibility": feasibility,
            "confidence": confidence,
            "reason": reason,
            "closest_latitude": closest_row['latitude'],
            "closest_longitude": closest_row['longitude']
        })
    except Exception as e:
        # Handle unexpected errors
        return jsonify({
            "feasibility": 0,
            "confidence": 0,
            "reason": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)