from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import re
import os
import numpy as np
import pandas as pd
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# ------------------------- Ticket Classifier -------------------------
class TicketClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_model()
        self.load_label_map()
    
    def load_model(self):
        try:
            print("Loading BERT model...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def load_label_map(self):
        try:
            label_map_path = os.path.join(self.model_path, 'label_map.json')
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print("Label map loaded:", self.label_map)
        except Exception as e:
            print(f"Error loading label map: {str(e)}")
            # Fallback label map
            self.label_map = {
                "0": "Change",
                "1": "Incident", 
                "2": "Problem",
                "3": "Request"
            }
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\-]', '', text)
        return text
    
    def predict(self, text):
        try:
            processed_text = self.preprocess_text(text)
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            results = {self.label_map[str(idx)].lower(): float(prob) 
                       for idx, prob in enumerate(probabilities)}
            predicted_idx = np.argmax(probabilities)
            predicted_label = self.label_map[str(predicted_idx)]
            confidence = float(probabilities[predicted_idx])
            return {
                'predicted_category': predicted_label.lower(),
                'confidence': confidence,
                'all_probabilities': results
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'predicted_category': 'error',
                'confidence': 0.0,
                'all_probabilities': {}
            }

# ------------------------- Initialize Model -------------------------
MODEL_PATH = "./model"  # Update to your model path
classifier = None

def initialize_model():
    global classifier
    try:
        classifier = TicketClassifier(MODEL_PATH)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        classifier = None

# ------------------------- Routes -------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_ticket():
    """Single ticket text classification"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        data = request.get_json()
        ticket_text = data.get('ticket_text', '').strip()
        if not ticket_text or len(ticket_text) < 5:
            return jsonify({'error': 'Ticket text too short', 'success': False}), 400
        
        result = classifier.predict(ticket_text)
        return jsonify({
            'success': True,
            'prediction': result['predicted_category'],
            'confidence': round(result['confidence'] * 100, 1),
            'all_probabilities': {k: round(v*100,1) for k,v in result['all_probabilities'].items()}
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """Batch classification from JSON list"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        data = request.get_json()
        tickets = data.get('tickets', [])
        if not tickets or not isinstance(tickets, list):
            return jsonify({'error': 'Please provide a list of tickets', 'success': False}), 400
        
        results = []
        for ticket_text in tickets:
            ticket_text = str(ticket_text).strip()
            if len(ticket_text) < 5:
                prediction = 'too_short'
                confidence = 0.0
            else:
                pred_result = classifier.predict(ticket_text)
                prediction = pred_result['predicted_category']
                confidence = float(pred_result['confidence'])
            results.append({'ticket_text': ticket_text, 'prediction': prediction, 'confidence': confidence})
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ------------------------- Updated classify_csv -------------------------
@app.route('/classify_csv', methods=['POST'])
def classify_csv():
    """
    Upload CSV, classify each row, return JSON array 
    with ticket_text, prediction, and email
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        if classifier is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # ✅ Ensure required columns exist
        if 'ticket_text' not in df.columns or 'email' not in df.columns:
            return jsonify({'error': 'CSV must have "ticket_text" and "email" columns'}), 400
        
        # Classify each row
        results = []
        for _, row in df.iterrows():
            ticket_text_str = str(row['ticket_text']).strip()
            email_str = str(row['email']).strip()
            if len(ticket_text_str) < 5:
                prediction = 'too_short'
            else:
                prediction = classifier.predict(ticket_text_str)['predicted_category']
            results.append({
            	'email': email_str,
                'ticket_text': ticket_text_str,
                'prediction': prediction
                
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if classifier is not None else "not_loaded"
    return jsonify({'status': 'healthy', 'model_status': model_status, 'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))})

# ------------------------- Main -------------------------
if __name__ == '__main__':
    initialize_model()
    print("Starting Flask server...")
    print("Access via LAN: http://192.168.221.101:5000/")
    print("Make sure model files are in './model' directory")
    # Use debug=False for LAN access and disable auto-reload
    app.run(debug=False, host='0.0.0.0', port=5000)

