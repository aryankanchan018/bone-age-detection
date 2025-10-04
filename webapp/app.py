from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import os
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BoneAgeApp")

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Model loading placeholders
# MODEL_PATH = '../saved_models/best_bone_age_model.keras'
# model = None


def allowed_file(filename):
    """Check if the uploaded file has an allowed image extension."""
    return (
        '.' in filename and 
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Temporarily disable prediction until model is ready
    return jsonify({'error': 'Model not loaded yet. Please try again later.'}), 503
    
    # Uncomment below once model is ready
    
    # if model is None:
    #     logger.error("Model not loaded. Cannot predict.")
    #     return jsonify({'error': 'Model not available. Please contact administrator.'}), 503
    #
    # if 'file' not in request.files:
    #     logger.warning("No file key in request.")
    #     return jsonify({'error': 'No file uploaded'}), 400
    # file = request.files['file']
    # if file.filename == '':
    #     logger.warning("Received empty filename.")
    #     return jsonify({'error': 'No file selected'}), 400
    # if not allowed_file(file.filename):
    #     logger.warning(f"Invalid file type: {file.filename}")
    #     return jsonify({'error': 'Invalid file type.'}), 400
    #
    # try:
    #     filename = secure_filename(file.filename)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     unique_filename = f"{timestamp}_{filename}"
    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    #     file.save(filepath)
    #     logger.info(f"Saved and processing: {unique_filename}")
    #
    #     img = image.load_img(filepath, target_size=(256, 256))
    #     img_array = image.img_to_array(img)
    #     img_array = np.expand_dims(img_array, axis=0)
    #     img_array = img_array / 255.0
    #     img_array = preprocess_input(img_array)
    #
    #     pred = model.predict(img_array, verbose=0)
    #     predicted_age = float(pred[0][0])
    #     predicted_years = predicted_age / 12
    #
    #     confidence = 'High' if 24 <= predicted_age <= 216 else 'Medium' if 12 <= predicted_age <= 228 else 'Low'
    #
    #     os.remove(filepath)
    #     logger.info(f"Deleted temp file: {unique_filename}")
    #
    #     return jsonify({
    #         'success': True,
    #         'predicted_age': round(predicted_age, 1),
    #         'predicted_years': round(predicted_years, 1),
    #         'confidence': confidence,
    #         'timestamp': datetime.now().isoformat()
    #     }), 200
    #
    # except Exception as e:
    #     logger.error(f"Prediction failed: {str(e)}", exc_info=True)
    #     if 'filepath' in locals() and os.path.exists(filepath):
    #         try:
    #             os.remove(filepath)
    #         except Exception:
    #             pass
    #     return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': False,
        'timestamp': datetime.now().isoformat(),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    }), 200


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
