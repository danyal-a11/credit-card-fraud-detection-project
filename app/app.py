"""
Credit Card Fraud Detection API
================================
Flask application serving the XGBoost fraud detection model.

Author: Danyal
Project: Credit Card Fraud Detection Portfolio Project
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
from datetime import datetime

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_artifacts():
    """
    Load all required model artifacts for prediction.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, '..', 'models')
    
    print("=" * 60)
    print("LOADING FRAUD DETECTION MODEL")
    print("=" * 60)
    
    # Load the deployment model
    model_path = os.path.join(models_path, 'fraud_model_deployment.pkl')
    print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    # Load metadata
    import json
    metadata_path = os.path.join(models_path, 'deployment_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model AUC: {metadata.get('test_performance', {}).get('auc', 'N/A')}")
    print(f"Fraud Recall: {metadata.get('test_performance', {}).get('fraud_recall', 'N/A')}")
    print("\n" + "=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 60 + "\n")
    
    return model, metadata


# Load model at startup
try:
    MODEL, MODEL_METADATA = load_model_artifacts()
    MODEL_LOADED = True
except Exception as e:
    print(f"ERROR loading model: {e}")
    MODEL_LOADED = False
    MODEL = None
    MODEL_METADATA = {}

# ============================================================================
# FEATURE SCALING PARAMETERS
# ============================================================================
# These values come from the original dataset statistics
# Used to scale Time and Amount features before prediction

TIME_MEAN = 94813.86
TIME_STD = 47488.15
AMOUNT_MEAN = 88.35
AMOUNT_STD = 250.12


def scale_time(time_value):
    """Scale time using StandardScaler parameters"""
    return (time_value - TIME_MEAN) / TIME_STD


def scale_amount(amount_value):
    """Scale amount using StandardScaler parameters"""
    return (amount_value - AMOUNT_MEAN) / AMOUNT_STD

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(probability):
    """
    Determine risk level based on fraud probability.
    """
    if probability > 0.8:
        return 'Critical', '#e74c3c'
    elif probability > 0.6:
        return 'High', '#e67e22'
    elif probability > 0.4:
        return 'Medium', '#f39c12'
    else:
        return 'Low', '#27ae60'


def get_recommendation(probability, amount):
    """
    Generate actionable recommendation based on fraud probability.
    """
    pct = probability * 100
    
    if probability > 0.8:
        return (
            f"BLOCK TRANSACTION - Critical fraud risk ({pct:.1f}%). "
            f"Immediate investigation required for ${amount:.2f} transaction."
        )
    elif probability > 0.6:
        return (
            f"HOLD FOR REVIEW - High fraud risk ({pct:.1f}%). "
            f"Transaction of ${amount:.2f} requires manual verification."
        )
    elif probability > 0.4:
        return (
            f"FLAG FOR MONITORING - Medium risk ({pct:.1f}%). "
            f"Add ${amount:.2f} transaction to watch list."
        )
    else:
        return (
            f"APPROVE - Low fraud risk ({pct:.1f}%). "
            f"Transaction of ${amount:.2f} appears legitimate."
        )


def validate_input(data):
    """
    Validate input data for prediction.
    """
    required_fields = ['time', 'amount'] + [f'V{i}' for i in range(1, 29)]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    try:
        time_val = float(data['time'])
        amount_val = float(data['amount'])
        
        if time_val < 0:
            return False, "Time must be non-negative"
        if amount_val < 0:
            return False, "Amount must be non-negative"
            
        for i in range(1, 29):
            float(data[f'V{i}'])
            
    except (ValueError, TypeError) as e:
        return False, f"Invalid numeric value: {e}"
    
    return True, None

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page - renders the fraud detection web interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for fraud detection.
    """
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'status': 'error'
        }), 503
    
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
        
        # Extract features
        time_val = float(data['time'])
        amount_val = float(data['amount'])
        v_features = [float(data[f'V{i}']) for i in range(1, 29)]
        
        # Scale time and amount
        time_scaled = scale_time(time_val)
        amount_scaled = scale_amount(amount_val)
        
        # Construct feature array: V1-V28, Time_scaled, Amount_scaled
        features = v_features + [time_scaled, amount_scaled]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        fraud_probability = MODEL.predict_proba(features_array)[0][1]
        is_fraud = fraud_probability > 0.5
        
        # Get risk assessment
        risk_level, color = get_risk_level(fraud_probability)
        recommendation = get_recommendation(fraud_probability, amount_val)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            'status': 'success',
            'fraud_probability': round(float(fraud_probability), 4),
            'is_fraud': bool(is_fraud),
            'risk_level': risk_level,
            'color': color,
            'recommendation': recommendation,
            'processing_time_ms': round(processing_time, 2),
            'transaction_details': {
                'amount': amount_val,
                'time': time_val
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'model_type': type(MODEL).__name__ if MODEL else None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get model information."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 503
    
    return jsonify({
        'status': 'success',
        'model_info': {
            'type': type(MODEL).__name__,
            'metadata': MODEL_METADATA
        }
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['GET /', 'POST /predict', 'GET /health', 'GET /model-info']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("\nModel Performance:")
    print(f"   AUC Score: {MODEL_METADATA.get('test_performance', {}).get('auc', 'N/A')}")
    print(f"   Fraud Recall: {MODEL_METADATA.get('test_performance', {}).get('fraud_recall', 'N/A')}")
    print("\nAPI Endpoints:")
    print("   GET  /           - Web interface")
    print("   POST /predict    - Single prediction")
    print("   GET  /health     - Health check")
    print("   GET  /model-info - Model details")
    print("\nOpen your browser and go to:")
    print("   http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)