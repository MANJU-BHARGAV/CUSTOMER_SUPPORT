from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import re
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class TicketClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {self.device}")
        
        # Load model and label map
        self.load_model()
        self.load_label_map()
    
    def load_model(self):
        """Load the saved BERT model and tokenizer"""
        try:
            print("🚀 Loading BERT model...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e
    
    def load_label_map(self):
        """Load label mapping from JSON file"""
        try:
            label_map_path = os.path.join(self.model_path, 'label_map.json')
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print("✅ Label map loaded:", self.label_map)
        except Exception as e:
            print(f"⚠️ Error loading label map: {str(e)} — using fallback mapping")
            # Fallback mapping (ensure order matches your training)
            self.label_map = {
                "0": "Change",
                "1": "Incident", 
                "2": "Problem",
                "3": "Request"
            }
    
    def preprocess_text(self, text):
        """Apply preprocessing similar to training"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\-]', '', text)
        return text
    
    def predict(self, text):
        """Predict category for input text"""
        try:
            processed_text = self.preprocess_text(text)
            print(f"📝 Original: {text[:100]}...")
            print(f"🧹 Processed: {processed_text[:100]}...")
            
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            results = {self.label_map[str(i)].lower(): float(prob) for i, prob in enumerate(probabilities)}
            predicted_idx = int(np.argmax(probabilities))
            predicted_label = self.label_map[str(predicted_idx)]
            confidence = float(probabilities[predicted_idx])
            
            return {
                'predicted_category': predicted_label.lower(),
                'confidence': confidence,
                'all_probabilities': results
            }
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            raise e


# ---------------------- Flask App Setup ----------------------

MODEL_PATH = "./model"
classifier = None

def initialize_model():
    """Load model when the app starts"""
    global classifier
    try:
        classifier = TicketClassifier(MODEL_PATH)
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize model: {str(e)}")
        classifier = None


@app.route('/')
def home():
    """Serve frontend"""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_ticket():
    """Classify support ticket text"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        data = request.get_json()
        if not data or 'ticket_text' not in data:
            return jsonify({'error': 'Missing ticket_text in request', 'success': False}), 400
        
        ticket_text = data['ticket_text'].strip()
        if len(ticket_text) < 5:
            return jsonify({'error': 'Ticket text too short', 'success': False}), 400
        
        result = classifier.predict(ticket_text)
        return jsonify({
            'success': True,
            'prediction': result['predicted_category'],
            'confidence': round(result['confidence'] * 100, 1),
            'all_probabilities': {k: round(v * 100, 1) for k, v in result['all_probabilities'].items()}
        })
    except Exception as e:
        print(f"❌ Error in classify_ticket: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if classifier is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    })


# ---------------------- Main Entry Point ----------------------

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📦 Ensuring model files are in './model' directory")
    initialize_model()
    app.run(host='0.0.0.0', port=5000)  # No debug=True for production

