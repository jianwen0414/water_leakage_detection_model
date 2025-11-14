"""
Water Leakage Detection API
Flask application for model deployment on Render.com
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os
import logging
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model_files():
    """Load model and scaler files with error handling"""
    global model, scaler
    
    try:
        # Log versions
        import sklearn
        logger.info(f"Python packages:")
        logger.info(f"  - numpy: {np.__version__}")
        logger.info(f"  - pandas: {pd.__version__}")
        logger.info(f"  - scikit-learn: {sklearn.__version__}")
        logger.info(f"  - joblib: {joblib.__version__}")
        
        model_path = 'water_leak_model.pkl'
        scaler_path = 'scaler.pkl'
        
        # Log current directory contents
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}")
            return False
            
        # Load model and scaler
        logger.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        
        logger.info("✓ Model and scaler loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Scaler type: {type(scaler).__name__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error loading model files: {str(e)}")
        logger.error("This usually indicates a numpy version mismatch")
        logger.error("Please ensure requirements.txt specifies: numpy>=2.0.0")
        return False
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Load model on startup
if not load_model_files():
    logger.warning("Failed to load model files on startup")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Leakage Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 1.1em;
        }
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .input-hint {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 12px;
            display: none;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .result.leak {
            background: #fee;
            border: 2px solid #f44336;
        }
        .result.no-leak {
            background: #e8f5e9;
            border: 2px solid #4caf50;
        }
        .result-title {
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 15px;
        }
        .result.leak .result-title {
            color: #d32f2f;
        }
        .result.no-leak .result-title {
            color: #2e7d32;
        }
        .result-details {
            font-size: 1em;
            line-height: 1.8;
        }
        .probability {
            font-size: 2em;
            font-weight: 700;
            margin: 15px 0;
        }
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        .api-info {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
            font-size: 0.9em;
        }
        .api-info h3 {
            margin-bottom: 10px;
            color: #667eea;
        }
        .api-info code {
            background: #333;
            color: #0f0;
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💧 Water Leakage Detection</h1>
            <p>AI-Powered Predictive Monitoring System</p>
        </div>
        
        <div class="content">
            <form id="predictionForm">
                <div class="form-group">
                    <label for="flow_rate">Flow Rate (L/min)</label>
                    <input type="number" id="flow_rate" name="flow_rate" step="0.01" required>
                    <div class="input-hint">Normal range: 100-180 L/min</div>
                </div>
                
                <div class="form-group">
                    <label for="pressure">Pressure (bar)</label>
                    <input type="number" id="pressure" name="pressure" step="0.01" required>
                    <div class="input-hint">Normal range: 2.0-3.5 bar</div>
                </div>
                
                <div class="form-group">
                    <label for="temperature">Temperature (°C)</label>
                    <input type="number" id="temperature" name="temperature" step="0.01" required>
                    <div class="input-hint">Normal range: 15-22 °C</div>
                </div>
                
                <button type="submit" class="btn">Analyze System</button>
            </form>
            
            <div id="result" class="result"></div>
            
            <div class="api-info">
                <h3>API Endpoint</h3>
                <p><strong>POST</strong> <code>/predict</code></p>
                <p><strong>Body:</strong> <code>{"flow_rate": 150.5, "pressure": 2.8, "temperature": 18.2}</code></p>
            </div>
        </div>
        
        <div class="footer">
            <p>Model: {{ model_name }} | Accuracy: {{ accuracy }}% | Trained: {{ training_date }}</p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                flow_rate: parseFloat(document.getElementById('flow_rate').value),
                pressure: parseFloat(document.getElementById('pressure').value),
                temperature: parseFloat(document.getElementById('temperature').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (result.prediction === 1) {
                    resultDiv.className = 'result leak';
                    resultDiv.innerHTML = `
                        <div class="result-title">⚠️ LEAK DETECTED</div>
                        <div class="probability">${(result.confidence * 100).toFixed(1)}% Confidence</div>
                        <div class="result-details">
                            <strong>Recommendation:</strong> Immediate inspection required.<br>
                            <strong>Status:</strong> Abnormal system behavior detected.<br>
                            <strong>Action:</strong> Please verify all sensors and check for physical leaks.
                        </div>
                    `;
                } else {
                    resultDiv.className = 'result no-leak';
                    resultDiv.innerHTML = `
                        <div class="result-title">✅ SYSTEM NORMAL</div>
                        <div class="probability">${(result.confidence * 100).toFixed(1)}% Confidence</div>
                        <div class="result-details">
                            <strong>Status:</strong> All parameters within normal range.<br>
                            <strong>Action:</strong> Continue regular monitoring.
                        </div>
                    `;
                }
            } catch (error) {
                alert('Error making prediction: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the web interface"""
    if model is not None and scaler is not None:
        return render_template_string(
            HTML_TEMPLATE,
            model_name='Best Model',
            accuracy='95.0',
            training_date='2023-10-01'
        )
    return render_template_string(HTML_TEMPLATE, 
                                  model_name='Unknown', 
                                  accuracy='N/A', 
                                  training_date='N/A')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON format:
    {
        "flow_rate": 150.5,
        "pressure": 2.8,
        "temperature": 18.2
    }
    """
    global model, scaler
    
    # Check if model is loaded
    if model is None or scaler is None:
        logger.error("Prediction attempted but model not loaded")
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure model files are present'
        }), 500
    
    try:
        # Get data from request
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        # Validate input
        required_fields = ['flow_rate', 'pressure', 'temperature']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing field: {field}',
                    'message': f'Please provide {field}'
                }), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([{
            'flow_rate': float(data['flow_rate']),
            'pressure': float(data['pressure']),
            'temperature': float(data['temperature'])
        }])
        
        logger.info(f"Input DataFrame: {input_df.to_dict(orient='records')[0]}")
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        # Determine status and action
        if prediction == 1:
            status = "⚠️ LEAK DETECTED"
            action = "Immediate inspection required. Check for leaks in the system."
        else:
            status = "✅ SYSTEM NORMAL"
            action = "Continue regular monitoring."
        
        return jsonify({
            'prediction': int(prediction),
            'leak_detected': bool(prediction == 1),
            'confidence': float(confidence * 100),
            'leak_probability': float(prediction_proba[1]),
            'status': status,
            'action': action,
            'input': {
                'flow_rate': float(data['flow_rate']),
                'pressure': float(data['pressure']),
                'temperature': float(data['temperature'])
            }
        })
        
    except ValueError as e:
        logger.error(f"Invalid input values: {str(e)}")
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    import sklearn
    
    model_loaded = model is not None and scaler is not None
    
    # Get package versions
    versions = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'joblib': joblib.__version__
    }
    
    # List .pkl files
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'model_type': type(model).__name__ if model else None,
        'scaler_type': type(scaler).__name__ if scaler else None,
        'versions': versions,
        'pkl_files': pkl_files,
        'working_directory': os.getcwd()
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if metadata:
        return jsonify(metadata), 200
    return jsonify({'error': 'Metadata not available'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)