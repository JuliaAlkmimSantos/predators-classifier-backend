
# app.py
import io
from flask import Flask, request, jsonify
from PIL import Image
from models_load import classify_image  # import your function

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Australian Wildlife Classifier API!"

@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Expects a file uploaded with form key 'image'.
    Returns JSON with predicted class and probabilities.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        # Convert uploaded file to PIL image
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image. {str(e)}"}), 400

    # Use your model_utils function
    pred_class, probs = classify_image(img)

    # Format probabilities as a dict {class_name: probability}
    prob_dict = {cls: float(prob) for cls, prob in zip(classify_image.classes, probs)}

   
