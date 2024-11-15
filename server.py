from flask import Flask, render_template, jsonify
import json
from pathlib import Path
import os

app = Flask(__name__)

# Global variable to track training status
training_status = {
    'is_complete': False
}

def clear_logs():
    """Clear previous training and test logs"""
    global training_status
    if os.path.exists('training_data.json'):
        os.remove('training_data.json')
    if os.path.exists('test_results.json'):
        os.remove('test_results.json')
    training_status['is_complete'] = False
    print("Previous logs cleared")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_training_data')
def get_training_data():
    try:
        with open('training_data.json', 'r') as f:
            data = json.load(f)
        # Add training status to response
        data['is_training_complete'] = training_status['is_complete']
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({
            'train_loss': [], 
            'val_loss': [], 
            'train_acc': [], 
            'val_acc': [],
            'is_training_complete': False
        })

@app.route('/set_training_complete')
def set_training_complete():
    global training_status
    training_status['is_complete'] = True
    return jsonify({'status': 'success'})

@app.route('/get_test_results')
def get_test_results():
    try:
        with open('test_results.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'images': [], 'predictions': [], 'labels': []})

if __name__ == '__main__':
    clear_logs()
    app.run(debug=True) 