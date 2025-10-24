"""
Scalp Hair Disease Detection System - Flask Backend
Author: Your Name
Date: October 22, 2025
Framework: Flask + YOLOv11
Purpose: Big Data Analytics using AI Project - REST API
"""

from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from datetime import datetime
import json

# ==================== FLASK APP CONFIGURATION ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ==================== LOAD YOLO MODEL ====================
model = None

def load_model():
    """Load YOLOv11 model on startup"""
    global model
    if model is None:
        try:
            model = YOLO("runs/detect/train7/weights/best.pt")
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
            return False
    return True

# ==================== DISEASE INFORMATION DATABASE ====================
DISEASE_INFO = {
    'Alopecia Areata': {
        'description': 'Autoimmune disorder causing patchy hair loss',
        'symptoms': 'Circular bald patches, nail changes, smooth areas',
        'treatment': 'Corticosteroid injections, topical immunotherapy, JAK inhibitors',
        'severity': 'Moderate to Severe',
        'specialist': 'Dermatologist, Immunologist'
    },
    'Contact Dermatitis': {
        'description': 'Skin inflammation from irritants or allergens',
        'symptoms': 'Redness, itching, scaling, burning sensation',
        'treatment': 'Avoid irritants, topical corticosteroids, antihistamines',
        'severity': 'Mild to Moderate',
        'specialist': 'Dermatologist, Allergist'
    },
    'Folliculitis': {
        'description': 'Inflammation of hair follicles',
        'symptoms': 'Small red bumps, pustules, itching, tenderness',
        'treatment': 'Antibacterial shampoos, topical/oral antibiotics',
        'severity': 'Mild to Moderate',
        'specialist': 'Dermatologist'
    },
    'Head_Lice': {
        'description': 'Parasitic infestation of the scalp',
        'symptoms': 'Intense itching, visible nits, scalp irritation',
        'treatment': 'Medicated shampoos, manual removal, oral ivermectin',
        'severity': 'Mild',
        'specialist': 'Primary Care Physician'
    },
    'Psoriasis': {
        'description': 'Chronic autoimmune skin condition',
        'symptoms': 'Silver-white scales, red patches, itching',
        'treatment': 'Coal tar shampoo, salicylic acid, biologics',
        'severity': 'Moderate to Severe',
        'specialist': 'Dermatologist, Rheumatologist'
    },
    'dandruff': {
        'description': 'Common scalp condition with flaking',
        'symptoms': 'White/yellow flakes, itchy scalp',
        'treatment': 'Anti-dandruff shampoos with zinc pyrithione',
        'severity': 'Mild',
        'specialist': 'Dermatologist'
    },
    'dry hair': {
        'description': 'Lack of moisture in hair strands',
        'symptoms': 'Brittle, dull hair, split ends, frizz',
        'treatment': 'Deep conditioning, coconut oil, reduce heat',
        'severity': 'Mild',
        'specialist': 'Trichologist'
    },
    'grey hair': {
        'description': 'Natural aging process of pigmentation loss',
        'symptoms': 'Gray or white hair strands',
        'treatment': 'Hair dye (cosmetic) or embrace natural look',
        'severity': 'Normal Aging',
        'specialist': 'None required'
    },
    'low hair density': {
        'description': 'Reduced number of hair follicles',
        'symptoms': 'Thin-looking hair, visible scalp',
        'treatment': 'Minoxidil, finasteride, PRP therapy, transplant',
        'severity': 'Mild to Moderate',
        'specialist': 'Trichologist, Hair Transplant Surgeon'
    }
}

# ==================== UTILITY FUNCTIONS ====================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    try:
        buffered = BytesIO()
        Image.fromarray(image_array).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', diseases=DISEASE_INFO.keys())

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, PNG, or BMP'}), 400
    
    try:
        # Load model if not already loaded
        if not load_model():
            return jsonify({'error': 'Model not available'}), 500
        
        # Save uploaded file with timestamp
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get confidence threshold from request
        conf_threshold = float(request.form.get('confidence', 0.25))
        
        # Read image
        image = Image.open(filepath)
        img_array = np.array(image)
        
        # Run YOLO prediction
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            label = model.names[cls_id]
            
            detection_data = {
                'disease': label,
                'confidence': round(conf * 100, 2),
                'info': DISEASE_INFO.get(label, {
                    'description': 'Information not available',
                    'symptoms': 'N/A',
                    'treatment': 'Consult a healthcare professional',
                    'severity': 'Unknown',
                    'specialist': 'Healthcare Professional'
                })
            }
            detections.append(detection_data)
        
        # Convert annotated image to base64
        result_img_base64 = image_to_base64(annotated_img)
        
        if result_img_base64 is None:
            return jsonify({'error': 'Failed to process result image'}), 500
        
        # Prepare response
        response = {
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'result_image': result_img_base64,
            'original_filename': file.filename,
            'timestamp': timestamp
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/disease_info/<disease_name>')
def disease_info(disease_name):
    """Get information about a specific disease"""
    info = DISEASE_INFO.get(disease_name)
    if info:
        return jsonify({
            'success': True,
            'disease': disease_name,
            **info
        })
    return jsonify({
        'success': False,
        'error': 'Disease not found'
    }), 404

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'num_classes': 9,
        'detectable_diseases': list(DISEASE_INFO.keys())
    })

@app.route('/api/diseases')
def get_diseases():
    """Get list of all detectable diseases"""
    return jsonify({
        'success': True,
        'diseases': list(DISEASE_INFO.keys()),
        'count': len(DISEASE_INFO)
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*80)
    print("🚀 SCALP HAIR DISEASE DETECTION - FLASK SERVER")
    print("="*80)
    print("\n🔄 Loading YOLO model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"✅ Detectable diseases: {len(DISEASE_INFO)}")
        print("\n" + "="*80)
        print("🌐 Starting Flask server...")
        print("📍 Access at: http://localhost:5000")
        print("📍 Network: http://YOUR_LOCAL_IP:5000")
        print("⚠️  Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model!")
        print("Please ensure 'runs/detect/train7/weights/best.pt' exists")
        print("="*80)
