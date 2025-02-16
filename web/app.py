from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained models using absolute paths
rf_model_projects = joblib.load(os.path.join(BASE_DIR, 'rf_model_project_delay.pkl'))
rf_model_clients = joblib.load(os.path.join(BASE_DIR, 'rf_model_client_satisfaction.pkl'))


# Error rates from previous results (update as needed)
project_delay_mse = 3.9339
project_delay_r2 = 0.9712
client_satisfaction_mse = 8.4795
client_satisfaction_r2 = -0.0422

# Function to predict project delay
def predict_project_delay(input_data_list):
    input_df = pd.DataFrame(input_data_list)  # Convert list of dicts to DataFrame
    
    expected_columns_projects = [
        "Bugs Found", "Bugs Fixed", "Bug Fix Ratio", "Team Size", "Hours Worked", 
        "Budget (USD)", "Risk Factor", "Critical Issues Found", "Critical Issue Ratio", 
        "Feature Delivery Ratio", "Team Skill Level", "Resource Utilization Rate", 
        "Client Involvement", "Complexity Score"
    ]
    
    input_df = input_df[expected_columns_projects]
    
    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_columns_projects = ["Team Skill Level", "Client Involvement", "Complexity Score"]
    for col in categorical_columns_projects:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    return rf_model_projects.predict(input_df).tolist()  # Return as list

def predict_client_satisfaction(input_data_list):
    input_df = pd.DataFrame(input_data_list)

    expected_columns_clients = [
        "Complaints Raised", "Complaints Resolved", "Complaint Resolution Ratio", "Resolution Time (hours)", "Feedback Sentiment",
        "Revenue Generated (USD)", "Service Renewal Likelihood (%)", "Client Communication Rating", 
        "Past Renewals", "Churn Risk"
    ]
    
    input_df = input_df[expected_columns_clients]

    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_columns_clients = ["Feedback Sentiment", "Churn Risk"]
    for col in categorical_columns_clients:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    return rf_model_clients.predict(input_df).tolist()  # Return as list

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.json 
        if not isinstance(data, list): 
            return jsonify({"error": "Input should be a list of JSON objects"}), 400

        project_delays = predict_project_delay(data)
        # client_satisfactions = predict_client_satisfaction(data)

        response = []
        for i in range(len(data)):
            response.append({
                "predicted_project_delay": project_delays[i],
                "error_rates": {
                    "project_delay_mse": project_delay_mse,
                    "project_delay_r2": project_delay_r2,
                    "client_satisfaction_mse": client_satisfaction_mse,
                    "client_satisfaction_r2": client_satisfaction_r2
                }
            })
        
        return jsonify(response)

    # If GET request, return the HTML page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
