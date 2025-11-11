from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model, scaler, and feature columns
print("Loading model artifacts...")
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
    
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("✓ Feature columns loaded successfully")
    print(f"  Total features: {len(feature_columns)}")
    
except Exception as e:
    print(f"ERROR loading model artifacts: {str(e)}")
    raise

def engineer_features(df_input):
    """
    Apply the same feature engineering as in training
    This function replicates the feature engineering from your notebook
    """
    df = df_input.copy()
    
    # Convert timestamp if provided
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        # Use current timestamp if not provided
        df['Timestamp'] = pd.Timestamp.now()
    
    # Temporal features
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['DayOfMonth'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    
    # Cyclical encoding for time features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
    
    # Sensor ID encoding (simple numeric encoding for single prediction)
    if 'Sensor_ID' in df.columns:
        # Simple hash encoding for sensor ID
        df['Sensor_ID_encoded'] = df['Sensor_ID'].apply(lambda x: hash(x) % 1000)
    else:
        df['Sensor_ID_encoded'] = 0
    
    # Domain-specific features
    df['Pressure_Flow_ratio'] = df['Pressure (bar)'] / (df['Flow Rate (L/s)'] + 1e-6)
    df['Temp_Pressure_product'] = df['Temperature (°C)'] * df['Pressure (bar)']
    df['Flow_Temp_ratio'] = df['Flow Rate (L/s)'] / (df['Temperature (°C)'] + 1e-6)
    
    # For rolling and lag features, use default values for single prediction
    # In production, you'd maintain a buffer of recent readings per sensor
    base_cols = ['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']
    
    for col in base_cols:
        # Rolling features - use current values as defaults
        df[f'{col}_rolling_mean'] = df[col]
        df[f'{col}_rolling_std'] = 0.0
        df[f'{col}_rolling_min'] = df[col]
        df[f'{col}_rolling_max'] = df[col]
        
        # Lag features - use current values as defaults
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col]
        
        # Rate of change - use 0 as default for single prediction
        df[f'{col}_rate_of_change'] = 0.0
    
    return df

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Water Leakage Detection API is running',
        'version': '1.0',
        'model_type': type(model).__name__,
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Make prediction for water leakage/anomaly detection',
            '/predict/batch': 'POST - Make predictions for multiple readings',
            '/features': 'GET - Get required feature list'
        },
        'required_input_fields': [
            'Pressure (bar)',
            'Flow Rate (L/s)',
            'Temperature (°C)'
        ],
        'optional_input_fields': [
            'Timestamp',
            'Sensor_ID'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_count': len(feature_columns)
    }), 200

@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of features used by the model"""
    return jsonify({
        'feature_count': len(feature_columns),
        'features': feature_columns,
        'required_input': [
            'Pressure (bar)',
            'Flow Rate (L/s)',
            'Temperature (°C)'
        ],
        'optional_input': [
            'Timestamp',
            'Sensor_ID'
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict anomaly for a single water sensor reading
    
    Expected JSON format:
    {
        "Pressure (bar)": 3.5,
        "Flow Rate (L/s)": 150.0,
        "Temperature (°C)": 20.0,
        "Timestamp": "2025-01-15 14:30:00",  // Optional
        "Sensor_ID": "SENSOR_001"  // Optional
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(df_engineered.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing engineered features: {list(missing_features)}',
                'note': 'This should not happen - contact administrator'
            }), 500
        
        # Select and order features according to training
        X = df_engineered[feature_columns]
        
        # Handle any infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Get probability if available
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Anomaly Detected (Leak/Burst)' if prediction == 1 else 'Normal',
            'timestamp': datetime.now().isoformat(),
            'input_data': {
                'pressure': float(data['Pressure (bar)']),
                'flow_rate': float(data['Flow Rate (L/s)']),
                'temperature': float(data['Temperature (°C)'])
            }
        }
        
        # Add probability if model supports it
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_scaled)[0]
            response['probability'] = {
                'normal': float(probability[0]),
                'anomaly': float(probability[1])
            }
            response['confidence'] = float(max(probability))
        
        # Add sensor info if provided
        if 'Sensor_ID' in data:
            response['sensor_id'] = data['Sensor_ID']
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict anomalies for multiple water sensor readings
    
    Expected JSON format:
    {
        "readings": [
            {
                "Pressure (bar)": 3.5,
                "Flow Rate (L/s)": 150.0,
                "Temperature (°C)": 20.0,
                "Sensor_ID": "SENSOR_001"
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'readings' not in data:
            return jsonify({'error': 'No readings provided. Expected format: {"readings": [...]}''}), 400
        
        readings = data['readings']
        
        if not isinstance(readings, list) or len(readings) == 0:
            return jsonify({'error': 'readings must be a non-empty list'}), 400
        
        # Validate required fields for all readings
        required_fields = ['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']
        
        results = []
        
        for idx, reading in enumerate(readings):
            missing_fields = [field for field in required_fields if field not in reading]
            
            if missing_fields:
                results.append({
                    'index': idx,
                    'error': f'Missing required fields: {missing_fields}',
                    'prediction': None
                })
                continue
            
            try:
                # Convert to DataFrame
                df = pd.DataFrame([reading])
                
                # Apply feature engineering
                df_engineered = engineer_features(df)
                
                # Select and order features
                X = df_engineered[feature_columns]
                
                # Handle any infinite or NaN values
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(0)
                
                # Scale the features
                X_scaled = scaler.transform(X)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                result = {
                    'index': idx,
                    'prediction': int(prediction),
                    'prediction_label': 'Anomaly Detected (Leak/Burst)' if prediction == 1 else 'Normal',
                    'input_data': {
                        'pressure': float(reading['Pressure (bar)']),
                        'flow_rate': float(reading['Flow Rate (L/s)']),
                        'temperature': float(reading['Temperature (°C)'])
                    }
                }
                
                # Add probability if available
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(X_scaled)[0]
                    result['probability'] = {
                        'normal': float(probability[0]),
                        'anomaly': float(probability[1])
                    }
                
                # Add sensor info if provided
                if 'Sensor_ID' in reading:
                    result['sensor_id'] = reading['Sensor_ID']
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e),
                    'prediction': None
                })
        
        # Summary statistics
        successful_predictions = [r for r in results if 'error' not in r]
        anomaly_count = sum(1 for r in successful_predictions if r['prediction'] == 1)
        
        return jsonify({
            'total_readings': len(readings),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(readings) - len(successful_predictions),
            'anomalies_detected': anomaly_count,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

if __name__ == '__main__':
    # Run on port 10000 (Render's default)
    app.run(host='0.0.0.0', port=10000, debug=False)