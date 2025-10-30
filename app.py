"""
EcoSort AI - Smart Waste Classification System
Flask Web Application with Enhanced Features

AI-powered waste classification using MobileNetV2 transfer learning
Features: Single upload, camera capture, batch processing
"""

import os
import warnings
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import base64
from io import BytesIO

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)

# Configure TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')

# Optimize TensorFlow for inference
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation

app = Flask(__name__)
app.secret_key = 'trashformer_secret_key_2024'
# Limit upload size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Configuration
DEBUG_MODE = False  # Set to False for production
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ANALYTICS_LOG = PROJECT_ROOT / "analytics_log.jsonl"

# Create upload directory
UPLOAD_FOLDER.mkdir(exist_ok=True)
ANALYTICS_LOG.touch(exist_ok=True)

# Class names (must match training order)
CLASS_NAMES = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']

# Class information
CLASS_INFO = {
    'cardboard': {'emoji': '📦', 'color': '#B07B3F', 'category': 'Recyclable', 'tips': [
        'Flatten boxes to save space',
        'Remove tape and labels when possible',
        'Keep dry and clean before recycling'
    ]},
    'e-waste': {'emoji': '🔌', 'color': '#FF6B3A', 'category': 'Hazardous', 'tips': [
        'Do not throw in general trash',
        'Take to certified e-waste collection centers',
        'Remove batteries and recycle separately'
    ]},
    'glass': {'emoji': '🍶', 'color': '#20C1C1', 'category': 'Recyclable', 'tips': [
        'Rinse and remove lids',
        'Sort by color if your facility requires',
        'Handle carefully to avoid breakage'
    ]},
    'medical': {'emoji': '🏥', 'color': '#D9534F', 'category': 'Biohazard', 'tips': [
        'Do not mix with household waste',
        'Use designated biohazard disposal bins',
        'Consult local guidelines for sharps and medicines'
    ]},
    'metal': {'emoji': '🥫', 'color': '#9AA2A6', 'category': 'Recyclable', 'tips': [
        'Rinse cans and remove labels',
        'Crush cans to save space',
        'Separate aluminum from steel if required'
    ]},
    'paper': {'emoji': '📄', 'color': '#E6D89C', 'category': 'Recyclable', 'tips': [
        'Keep paper clean and dry',
        'Remove staples and clips when possible',
        'Avoid recycling coated or greasy paper'
    ]},
    'plastic': {'emoji': '🔄', 'color': '#4C7BF5', 'category': 'Recyclable', 'tips': [
        'Check local rules for plastic types',
        'Rinse containers and remove caps',
        'Avoid single-use plastics when possible'
    ]}
}

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is not None:
        return model
    
    # Look for models in order of preference - prioritize finetuned models
    model_patterns = [
        'trashformer_finetuned_*.keras',
        'trashformer_best_*.keras',
        'trashformer_final.keras'
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = list(MODEL_DIR.glob(pattern))
        if matches:
            model_path = max(matches, key=lambda x: x.stat().st_ctime)
            break
    
    if model_path is None:
        raise Exception("No trained model found in models/ directory")
    
    model = tf.keras.models.load_model(str(model_path))
    return model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    """Make prediction on image"""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Create detailed results
    results = []
    for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        results.append({
            'class': class_name,
            'probability': float(prob),
            'emoji': CLASS_INFO[class_name]['emoji'],
            'color': CLASS_INFO[class_name]['color'],
            'category': CLASS_INFO[class_name]['category']
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return predicted_class, confidence, results

# Grad-CAM utilities for live localization
def _get_last_conv_layer(model_obj):
    """Try to find the last convolutional layer in the model."""
    # Prefer layers with 4D output (feature maps)
    for layer in reversed(model_obj.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def _compute_gradcam_heatmap(img_array: np.ndarray, model_obj, class_index: int):
    """Compute Grad-CAM heatmap for a single preprocessed image batch (1, H, W, 3)."""
    last_conv_name = _get_last_conv_layer(model_obj)
    if last_conv_name is None:
        return None

    last_conv_layer = model_obj.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model_obj.inputs],
        [last_conv_layer.output, model_obj.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            # Binary case; use the single logit/prob
            pred_score = predictions[:, 0]
        else:
            pred_score = predictions[:, class_index]
    grads = tape.gradient(pred_score, conv_outputs)

    if grads is None:
        return None

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Relu and normalize
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return None
    heatmap /= (max_val + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def _heatmap_to_bbox(heatmap: np.ndarray, threshold: float = 0.6):
    """Convert heatmap to a tight bounding box (normalized x,y,w,h) using a threshold.
    Returns None if no significant activation.
    """
    mask = heatmap >= threshold
    if not np.any(mask):
        # fallback: lower threshold
        threshold = 0.4
        mask = heatmap >= threshold
        if not np.any(mask):
            return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h, w = heatmap.shape
    # Normalize to 0-1
    x = float(x_min) / w
    y = float(y_min) / h
    bw = float(x_max - x_min + 1) / w
    bh = float(y_max - y_min + 1) / h
    return {
        'x': max(0.0, min(1.0, x)),
        'y': max(0.0, min(1.0, y)),
        'w': max(0.0, min(1.0, bw)),
        'h': max(0.0, min(1.0, bh)),
    }

def load_training_stats():
    """Load training statistics"""
    history_path = PROJECT_ROOT / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

# ========== Analytics logging helpers ==========
def _append_detection_log(entry: dict):
    try:
        with open(ANALYTICS_LOG, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

def _read_detection_logs():
    entries = []
    try:
        with open(ANALYTICS_LOG, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return entries

@app.route('/')
def index():
    """Main page"""
    # Load model
    try:
        model = load_model()
        model_loaded = True
        # Get the actual model name from the loaded model path
        model_patterns = [
            'trashformer_finetuned_*.keras',
            'trashformer_best_*.keras',
            'trashformer_final.keras'
        ]
        model_name = 'trashformer_finetuned_*.keras'
    except Exception as e:
        model_loaded = False
        model_name = None
    
    # Load training stats
    training_stats = load_training_stats()
    
    # Calculate training stats for template
    if training_stats:
        max_accuracy = max(training_stats['val_accuracy']) * 100
        total_epochs = len(training_stats['loss'])
    else:
        max_accuracy = 85.16  # Default value
        total_epochs = 29
    
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         model_name=model_name,
                         training_stats=training_stats,
                         max_accuracy=max_accuracy,
                         total_epochs=total_epochs,
                         class_names=CLASS_NAMES,
                         class_info=CLASS_INFO)

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/analytics/data')
def analytics_data():
    entries = _read_detection_logs()
    return jsonify({ 'entries': entries })

@app.route('/analytics/export.csv')
def analytics_export_csv():
    import csv
    from io import StringIO
    entries = _read_detection_logs()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['timestamp_iso', 'source', 'prediction', 'confidence', 'latitude', 'longitude'])
    for e in entries:
        ts = e.get('timestamp')
        ts_iso = datetime.utcfromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) else ''
        writer.writerow([
            ts_iso,
            e.get('source', ''),
            e.get('prediction', ''),
            e.get('confidence', ''),
            e.get('latitude', ''),
            e.get('longitude', '')
        ])
    output.seek(0)
    return output.getvalue(), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename="trashformer_analytics.csv"'
    }

@app.route('/log_geo', methods=['POST'])
def log_geo():
    try:
        data = request.get_json() or {}
        entry = {
            'timestamp': int(tf.timestamp().numpy()),
            'source': data.get('source', 'unknown'),
            'prediction': data.get('prediction'),
            'confidence': data.get('confidence'),
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude')
        }
        _append_detection_log(entry)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/export.pdf')
def analytics_export_pdf():
    try:
        # Lazy import to avoid hard dependency during runtime until used
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        entries = _read_detection_logs()
        # Compute counts
        counts = {}
        for e in entries:
            label = e.get('prediction')
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1

        # Sort top classes
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Prepare PDF
        from io import BytesIO
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        y = height - 20 * mm
        pdf.setTitle("Trashformer Analytics Report")
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(20 * mm, y, "Trashformer Analytics Report")
        y -= 10 * mm
        pdf.setFont("Helvetica", 11)
        pdf.drawString(20 * mm, y, f"Generated: {datetime.utcnow().isoformat()}Z")
        y -= 8 * mm
        pdf.drawString(20 * mm, y, f"Total Detections: {len(entries)}")

        # Counts table
        y -= 14 * mm
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(20 * mm, y, "Classification Counts by Category")
        y -= 8 * mm
        data = [["Category", "Count"]] + [[k.title(), v] for k, v in top]
        table = Table(data, colWidths=[80*mm, 30*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
            ('FONT', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONT', (0,1), (-1,-1), 'Helvetica'),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        # Render table
        w, h = table.wrapOn(pdf, width - 40*mm, y)
        table.drawOn(pdf, 20*mm, y - h)
        y = y - h - 12 * mm

        # Recent detections (up to 15)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(20 * mm, y, "Recent Detections")
        y -= 8 * mm
        recent = entries[-15:]
        data2 = [["Time (UTC)", "Source", "Prediction", "Conf."]]
        for e in recent:
            ts = e.get('timestamp')
            ts_iso = datetime.utcfromtimestamp(ts).isoformat()+'Z' if isinstance(ts, (int, float)) else ''
            data2.append([
                ts_iso,
                str(e.get('source', '')),
                str(e.get('prediction', '')),
                f"{round(float(e.get('confidence', 0))*100, 1)}%"
            ])
        table2 = Table(data2, colWidths=[55*mm, 25*mm, 40*mm, 20*mm])
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (3,1), (3,-1), 'RIGHT'),
            ('FONT', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONT', (0,1), (-1,-1), 'Helvetica'),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        w2, h2 = table2.wrapOn(pdf, width - 40*mm, y)
        # If not enough space, add a page
        if y - h2 < 20*mm:
            pdf.showPage()
            y = height - 20*mm
        table2.drawOn(pdf, 20*mm, y - h2)

        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        return buffer.getvalue(), 200, {
            'Content-Type': 'application/pdf',
            'Content-Disposition': 'attachment; filename="trashformer_analytics.pdf"'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single image prediction"""
    try:
        if DEBUG_MODE:
            print("=== Single Image Prediction Request ===")
            print("Files in request:", list(request.files.keys()))
        
        model = load_model()
        
        if 'image' not in request.files:
            if DEBUG_MODE:
                print("ERROR: No 'image' field in request files")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if DEBUG_MODE:
            print(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read image
            image = Image.open(file.stream)
            
            # Make prediction
            predicted_class, confidence, results = predict_image(image)
            
            # Convert image to base64 for display (optimized)
            buffered = BytesIO()
            # Resize large images for faster transmission
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            response_data = {
                'success': True,
                'prediction': predicted_class,
                'confidence': float(confidence),
                'results': results,
                'image_base64': img_str,
                'class_info': CLASS_INFO[predicted_class]
            }
            # Log detection (no location unless provided client-side later)
            _append_detection_log({
                'timestamp': int(tf.timestamp().numpy()),
                'source': 'upload',
                'prediction': predicted_class,
                'confidence': float(confidence),
                'all_results': results
            })
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"ERROR in predict(): {str(e)}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch image prediction"""
    try:
        if DEBUG_MODE:
            print("=== Batch Image Prediction Request ===")
        
        model = load_model()
        
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No images selected'}), 400
        
        results = []
        successful = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read image
                    image = Image.open(file.stream)
                    
                    # Make prediction
                    predicted_class, confidence, class_results = predict_image(image)
                    
                    # Convert image to base64 for display (optimized)
                    buffered = BytesIO()
                    max_size = (400, 400)  # Smaller for batch
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    image.save(buffered, format="JPEG", quality=75, optimize=True)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    results.append({
                        'filename': secure_filename(file.filename),
                        'prediction': predicted_class,
                        'confidence': float(confidence),
                        'image_base64': img_str,
                        'class_info': CLASS_INFO[predicted_class],
                        'all_results': class_results
                    })
                    successful += 1
                    
                except Exception as e:
                    results.append({
                        'filename': secure_filename(file.filename),
                        'error': str(e),
                        'prediction': 'Error',
                        'confidence': 0.0
                    })
        
        # Calculate statistics
        if successful > 0:
            confidences = [r['confidence'] for r in results if 'confidence' in r and r['confidence'] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            high_confidence = sum(1 for c in confidences if c > 0.8)
        else:
            avg_confidence = 0
            high_confidence = 0
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': {
                'total_images': len(results),
                'successful': successful,
                'avg_confidence': avg_confidence,
                'high_confidence': high_confidence
            }
        })
        
        # Log each successful detection in batch
        try:
            for r in results:
                if 'prediction' in r and 'confidence' in r:
                    _append_detection_log({
                        'timestamp': int(tf.timestamp().numpy()),
                        'source': 'batch',
                        'prediction': r.get('prediction'),
                        'confidence': float(r.get('confidence', 0)),
                        'all_results': r.get('all_results', [])
                    })
        except Exception:
            pass
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    """Handle camera capture prediction"""
    try:
        model = load_model()
        
        # Get base64 image data
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Make prediction
        predicted_class, confidence, results = predict_image(image)
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': float(confidence),
            'results': results,
            'image_base64': img_str,
            'class_info': CLASS_INFO[predicted_class]
        }
        _append_detection_log({
            'timestamp': int(tf.timestamp().numpy()),
            'source': 'camera',
            'prediction': predicted_class,
            'confidence': float(confidence),
            'all_results': results
        })
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_live', methods=['POST'])
def predict_live():
    """Handle live camera frame prediction (lightweight response)"""
    try:
        model = load_model()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image (data URL format)
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        predicted_class, confidence, results = predict_image(image)

        # Grad-CAM bbox
        try:
            img_array = preprocess_image(image)
            predicted_idx = np.argmax([r['probability'] for r in results])
            heatmap = _compute_gradcam_heatmap(img_array, model, predicted_idx)
            if heatmap is not None:
                bbox = _heatmap_to_bbox(heatmap, threshold=0.6)
            else:
                bbox = None
        except Exception:
            bbox = None

        # Lightweight response (omit echoing image)
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': float(confidence),
            'results': results,
            'class_info': CLASS_INFO[predicted_class]
        }
        _append_detection_log({
            'timestamp': int(tf.timestamp().numpy()),
            'source': 'live',
            'prediction': predicted_class,
            'confidence': float(confidence),
            'all_results': results
        })
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
