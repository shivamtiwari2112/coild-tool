# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, stream_with_context
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
import subprocess
import virtualenv
import datetime
import tempfile
import sys
import shutil
import time
import importlib.util
import logging
from asr_script import transcribe_audio
# Add these imports at the top of your file if needed
from datetime import datetime
import re
from io import StringIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads' # model upload folder
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['UPLOAD_AUDIO_FOLDER'] = 'audio_uploads'


# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_AUDIO_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'models'), exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'model_weights': {'h5', 'pth', 'pt', 'ckpt', 'pb', 'tflite'},
    'requirements': {'txt'},
    'inference': {'py'},
    'scripts': {'py', 'sh', 'json', 'yaml', 'yml'}
}

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

# Load data from a JSON file
def load_data():
    if os.path.exists('data/asr_data.json'):
        with open('data/asr_data.json', 'r') as f:
            return json.load(f)
    return {}

# Save data to JSON file
def save_data(data):
    os.makedirs('data', exist_ok=True)
    with open('data/asr_data.json', 'w') as f:
        json.dump(data, f, indent=2)



# Add this route to your app.py file
@app.route('/model_inference', methods=['GET', 'POST'])
def model_inference():
    data = load_data()
    
    # Get all available datasets
    datasets = list(data.keys())
    
    # Get current year for the form's default value
    current_year = datetime.now().year
    
    if request.method == 'POST':
        # Get form data
        dataset = request.form.get('dataset')
        language = request.form.get('language')
        model_name = request.form.get('model_name')
        year = request.form.get('year', current_year)
        paper_url = request.form.get('paper_url', '#')
        code_url = request.form.get('code_url', '#')
        extra_training_data = 'extra_training_data' in request.form
        
        # Get uploaded output file
        if 'output_file' not in request.files or not request.files['output_file'].filename:
            flash('Output file is required', 'danger')
            return render_template('model_inference.html', datasets=datasets, current_year=current_year)
        
        output_file = request.files['output_file']
        
        # Check file type and read content
        if not output_file.filename.endswith('.txt'):
            flash('Only .txt files are allowed', 'danger')
            return render_template('model_inference.html', datasets=datasets, current_year=current_year)
        
        # Read and process the output file
        output_text = output_file.read().decode('utf-8')
        
        # Compute WER using the ground truth for the selected dataset and language
        wer_result = compute_wer(dataset, language, output_text)
        
        # Generate a temporary model ID
        model_id = str(uuid.uuid4())
        
        # Get existing models for this dataset and language
        existing_models = []
        if dataset in data and language in data[dataset]:
            existing_models = data[dataset][language]
        
        # Create new model entry
        new_model = {
            "model": model_name,
            "test_wer": wer_result,
            "year": int(year) if year else current_year,
            "extra_training_data": extra_training_data,
            "paper": paper_url if paper_url else "#",
            "code": code_url if code_url else "#",
            "model_id": model_id,
            "is_new_submission": True  # Flag to highlight in the table
        }
        
        # Insert new model and sort all models by WER
        all_models = existing_models + [new_model]
        all_models = sorted(all_models, key=lambda x: float(x.get('test_wer', 100.0)))
        
        # Add rank to each model
        for i, model in enumerate(all_models):
            model['rank'] = i + 1
        
        # Create results summary
        results = {
            'model_id': model_id,
            'model_name': model_name,
            'test_wer': wer_result,
            'rank': new_model['rank'],
            'dataset': dataset,
            'language': language
        }
        
        # Store the new model in session for later addition to leaderboard if requested
        if 'temp_models' not in app.config:
            app.config['temp_models'] = {}
        app.config['temp_models'][model_id] = {
            'model': new_model,
            'dataset': dataset,
            'language': language
        }
        
        return render_template('model_inference.html', 
                              datasets=datasets,
                              languages=get_languages_for_dataset(dataset),
                              selected_dataset=dataset,
                              selected_language=language,
                              current_year=current_year,
                              results=results,
                              models=all_models)
    
    # GET request - just show the form
    return render_template('model_inference.html', 
                          datasets=datasets,
                          current_year=current_year)

# Route to save evaluated model to the leaderboard
@app.route('/save_to_leaderboard', methods=['POST'])
def save_to_leaderboard():
    model_id = request.form.get('model_id')
    
    if not model_id or 'temp_models' not in app.config or model_id not in app.config['temp_models']:
        flash('Model information not found', 'danger')
        return redirect('/model_inference')
    
    # Get model data from temporary storage
    temp_data = app.config['temp_models'][model_id]
    model = temp_data['model']
    dataset = temp_data['dataset']
    language = temp_data['language']
    
    # Remove the highlight flag before saving to leaderboard
    if 'is_new_submission' in model:
        del model['is_new_submission']
    
    # Load current data
    data = load_data()
    
    # Update the data structure
    if dataset not in data:
        data[dataset] = {}
    
    if language not in data[dataset]:
        data[dataset][language] = []
    
    # Add model to the appropriate dataset/language
    data[dataset][language].append(model)
    
    # Save the updated data
    save_data(data)
    
    # Clean up the temporary storage
    del app.config['temp_models'][model_id]
    
    flash(f'Model "{model["model"]}" has been added to the leaderboard', 'success')
    return redirect(f'/?dataset={dataset}&language={language}')


# Helper function to get languages for a specific dataset
def get_languages_for_dataset(dataset):
    data = load_data()
    if dataset in data:
        return list(data[dataset].keys())
    return []

# Helper function to compute WER
def compute_wer(dataset, language, model_output):
    """
    Compute Word Error Rate by comparing model output against ground truth.
    
    Args:
        dataset: The dataset name
        language: The language name
        model_output: The output text from the model to evaluate
        
    Returns:
        float: The calculated WER percentage
    """
    try:
        # Load ground truth data for the selected dataset and language
        # You would need to implement this based on your data structure
        # ground_truth = load_ground_truth(dataset, language)

        with open('gt.txt', 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        
        if not ground_truth:
            # If no ground truth available, return a default high WER
            return 99.9
        
        # Normalize both texts for comparison
        ground_truth_words = normalize_text(ground_truth).split()
        model_output_words = normalize_text(model_output).split()
        
        # Calculate WER
        # Simple implementation - you may want to use a more sophisticated approach
        if len(ground_truth_words) == 0:
            return 100.0
        
        # Calculate edit distance
        distance = levenshtein_distance(ground_truth_words, model_output_words)
        
        # Calculate WER as a percentage
        wer = (distance / len(ground_truth_words)) * 100
        
        # Round to one decimal place
        return round(wer, 1)
    
    except Exception as e:
        print(f"Error computing WER: {str(e)}")
        return 99.9

def normalize_text(text):
    """Normalize text for WER calculation"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def levenshtein_distance(list1, list2):
    """Calculate Levenshtein distance between two word lists"""
    # Create a table to store results of subproblems
    m, n = len(list1), len(list2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill dp[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first list is empty, insert all from list2
            if i == 0:
                dp[i][j] = j
            # If second list is empty, remove all from list1
            elif j == 0:
                dp[i][j] = i
            # If last characters are same, ignore last char
            elif list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            # If last characters are different, consider all possibilities
            else:
                dp[i][j] = 1 + min(dp[i][j-1],      # Insert
                                   dp[i-1][j],      # Remove
                                   dp[i-1][j-1])    # Replace
    
    return dp[m][n]

def load_ground_truth(dataset, language):
    """
    Load ground truth for a specific dataset and language.
    This is a placeholder - implement based on your data structure.
    """
    # Example implementation - replace with your actual data loading logic
    ground_truth_path = f"data/ground_truth/{dataset}/{language}/reference.txt"
    
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # If no file exists, return empty string
    return ""


@app.route('/create-environment', methods=['GET', 'POST'])
def create_environment():
    if request.method == 'GET':
        # For GET requests, load available models for the dropdown
        data = load_data()
        models = []
        
        # Extract all models from the data structure
        for dataset in data:
            for language in data[dataset]:
                for model in data[dataset][language]:
                    if 'model' in model and 'model_id' in model:
                        models.append({
                            'id': model['model_id'],
                            'name': f"{model['model']} ({dataset}/{language})"
                        })
        
        return render_template('create-environment.html', models=models)
    
    elif request.method == 'POST':
        # For POST requests, create the environment
        if request.content_type == 'application/json':
            # Handle AJAX JSON request
            data = request.json
        else:
            # Handle regular form submission (fallback)
            data = request.form
        
        environment_name = data.get('environment_name')
        python_version = data.get('python_version')
        selected_model_id = data.get('selected_model')
        custom_install_commands = data.get('custom_install_commands', {})
        
        # Validate required fields
        if not environment_name or not python_version or not selected_model_id:
            response = {
                'success': False,
                'message': 'Missing required fields'
            }
            return jsonify(response)
        
        try:
            # Create environment directory
            env_dir = os.path.join('environments', environment_name)
            os.makedirs(env_dir, exist_ok=True)
            
            # Create virtual environment
            virtualenv.create_environment(env_dir, site_packages=False)
            
            # Get model details and requirements file
            model_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models', selected_model_id)
            requirements_path = os.path.join(model_dir, 'requirements.txt')
            
            # Install dependencies from requirements.txt
            if os.path.exists(requirements_path):
                # Determine pip executable path based on OS
                pip_path = os.path.join(env_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(env_dir, 'Scripts', 'pip')
                
                # Install each dependency
                with open(requirements_path, 'r') as f:
                    for line in f:
                        dependency = line.strip()
                        if not dependency or dependency.startswith('#'):
                            continue
                            
                        # Parse package name
                        package_name = dependency.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        
                        # Check if there's a custom install command
                        if package_name in custom_install_commands and custom_install_commands[package_name]:
                            install_cmd = custom_install_commands[package_name]
                            subprocess.run(f"{pip_path} {install_cmd}", shell=True, check=True)
                        else:
                            subprocess.run([pip_path, 'install', dependency], check=True)
            
            # Save environment metadata
            metadata = {
                'name': environment_name,
                'python_version': python_version,
                'model_id': selected_model_id,
                'created_at': datetime.datetime.now().isoformat(),
            }
            
            with open(os.path.join(env_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            response = {
                'success': True,
                'message': f'Environment {environment_name} created successfully'
            }
            return jsonify(response)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            response = {
                'success': False,
                'message': str(e)
            }
            return jsonify(response)

@app.route('/api/model-dependencies/<model_id>')
def get_model_dependencies(model_id):
    """API endpoint to get a model's dependencies"""
    model_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model_id)
    requirements_path = os.path.join(model_dir, 'requirements.txt')
    
    dependencies = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse dependency information
                parts = line.split('==')
                if len(parts) == 2:
                    dependencies.append({
                        'name': parts[0].strip(),
                        'version': parts[1].strip()
                    })
                else:
                    # Handle other formats like >=, <=, etc.
                    name_parts = line.split('>=') if '>=' in line else (line.split('<=') if '<=' in line else [line, ''])
                    dependencies.append({
                        'name': name_parts[0].strip(),
                        'version': name_parts[1].strip() if len(name_parts) > 1 else None
                    })
    
    return jsonify({'dependencies': dependencies})


@app.route('/')
def index():
    data = load_data()
    
    # Get all available datasets
    datasets = list(data.keys())
    
    # Default to first dataset if available, or use query parameter
    selected_dataset = request.args.get('dataset', datasets[0] if datasets else None)
    
    # Initialize variables with defaults
    languages = []
    selected_language = None
    models = []
    
    if selected_dataset and selected_dataset in data:
        # Get all languages for the selected dataset
        languages = list(data[selected_dataset].keys())
        
        # Default to first language if available, or use query parameter
        selected_language = request.args.get('language', languages[0] if languages else None)
        
        if selected_language and selected_language in data[selected_dataset]:
            # Get models for the selected dataset and language
            models_data = data[selected_dataset][selected_language]
            
            # Ensure all models have the required fields
            for model in models_data:
                if 'test_wer' not in model:
                    # If 'test_wer' is missing but 'wer' exists, use that
                    if 'wer' in model:
                        model['test_wer'] = model['wer']
                    else:
                        # Otherwise set a default value
                        model['test_wer'] = 100.0  # Default high WER
            
            # Sort models by WER (ascending)
            models = sorted(models_data, key=lambda x: x.get('test_wer', 100.0))
            
            # Add rank to each model
            for i, model in enumerate(models):
                model['rank'] = i + 1
    
    return render_template('index.html', 
                          datasets=datasets,
                          languages=languages,
                          selected_dataset=selected_dataset,
                          selected_language=selected_language,
                          models=models)

@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    data = load_data()
    datasets = list(data.keys())
    
    if request.method == 'POST':
        # Get form data
        model_name = request.form.get('model_name')
        dataset = request.form.get('dataset')
        language = request.form.get('language')
        
        # Validate required fields
        if not model_name:
            flash('Model name is required', 'danger')
            return render_template('upload_model.html', datasets=datasets, message="Model name is required", message_type="danger")
        
        if not dataset or not language:
            flash('Dataset and language are required', 'danger')
            return render_template('upload_model.html', datasets=datasets, message="Dataset and language are required", message_type="danger")
        
        # Check required files
        if 'model_weights' not in request.files or not request.files['model_weights'].filename:
            return render_template('upload_model.html', datasets=datasets, message="Model weights file is required", message_type="danger")
        
        if 'requirements_file' not in request.files or not request.files['requirements_file'].filename:
            return render_template('upload_model.html', datasets=datasets, message="Requirements file is required", message_type="danger")
        
        if 'inference_file' not in request.files or not request.files['inference_file'].filename:
            return render_template('upload_model.html', datasets=datasets, message="Inference file is required", message_type="danger")
        
        # Get files
        model_weights = request.files['model_weights']
        requirements_file = request.files['requirements_file']
        inference_file = request.files['inference_file']
        
        # Validate file types
        if not allowed_file(model_weights.filename, 'model_weights'):
            return render_template('upload_model.html', datasets=datasets, message="Invalid model weights file format", message_type="danger")
        
        if not allowed_file(requirements_file.filename, 'requirements'):
            return render_template('upload_model.html', datasets=datasets, message="Invalid requirements file format", message_type="danger")
        
        if not allowed_file(inference_file.filename, 'inference'):
            return render_template('upload_model.html', datasets=datasets, message="Invalid inference file format", message_type="danger")
        
        # Check file names
        if requirements_file.filename != 'requirements.txt':
            return render_template('upload_model.html', datasets=datasets, message="Requirements file must be named 'requirements.txt'", message_type="danger")
        
        if inference_file.filename != 'model_inference.py':
            return render_template('upload_model.html', datasets=datasets, message="Inference file must be named 'model_inference.py'", message_type="danger")
        
        # Create a unique folder for this model
        model_id = str(uuid.uuid4())
        model_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save files
        model_weights.save(os.path.join(model_dir, secure_filename(model_weights.filename)))
        requirements_file.save(os.path.join(model_dir, 'requirements.txt'))
        inference_file.save(os.path.join(model_dir, 'model_inference.py'))
        
        # Save additional scripts if provided
        if 'model_scripts' in request.files:
            script_files = request.files.getlist('model_scripts')
            for script in script_files:
                if script.filename and allowed_file(script.filename, 'scripts'):
                    script.save(os.path.join(model_dir, secure_filename(script.filename)))
        
        # Get additional model details
        year = request.form.get('model_year', '')
        test_wer = request.form.get('test_wer', '')
        extra_training = 'extra_training_data' in request.form
        paper_url = request.form.get('paper_url', '')
        code_url = request.form.get('code_url', '')
        
        # Create model data object
        model_data = {
            "model": model_name,
            "test_wer": float(test_wer) if test_wer else 100.0,
            "year": int(year) if year else None,
            "extra_training_data": extra_training,
            "paper": paper_url,
            "code": code_url,
            "model_id": model_id
        }
        
        # Update the ASR data structure
        if dataset not in data:
            data[dataset] = {}
        
        if language not in data[dataset]:
            data[dataset][language] = []
        
        # Add model to the dataset/language
        data[dataset][language].append(model_data)
        
        # Save the updated data
        save_data(data)
        
        # Redirect with success query parameter
        return redirect(url_for('upload_model', success='true'))
    
    return render_template('upload_model.html', datasets=datasets)




@app.route('/api/data')
def get_data():
    data = load_data()
    dataset = request.args.get('dataset')
    language = request.args.get('language')
    
    if dataset in data and language in data[dataset]:
        return jsonify(data[dataset][language])
    return jsonify([])

@app.route('/api/datasets')
def get_datasets():
    data = load_data()
    return jsonify(list(data.keys()))

@app.route('/api/languages')
def get_languages():
    data = load_data()
    dataset = request.args.get('dataset')
    
    if dataset in data:
        return jsonify(list(data[dataset].keys()))
    return jsonify([])


# Add the new route for inference
@app.route('/upload_audio', methods=['GET', 'POST'])
def upload_audio():
    data = load_data()

    # Get available models
    models = []
    for dataset in data:
        for language in data[dataset]:
            for model in data[dataset][language]:
                if 'model' in model:
                    models.append({"id": model['model'], "name": model['model']})

    models = list({m['id']: m for m in models}.values())  # Remove duplicates

    if request.method == 'POST':
        model_id = request.form.get('model_id')
        language = request.form.get('language')
        files = request.files.getlist('audio_files')

        if not model_id or not files:
            flash("Model and audio files are required.", "danger")
            return render_template('upload_audio.html', models=models)

        saved_files = []
        results = []
        
        for file in files:
            if file.filename.endswith('.wav'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_AUDIO_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filepath)

                # Run ASR inference
                transcription = transcribe_audio(model_id, language, audio_file=filepath)  # Call the ASR function
                
                results.append({
                    "filename": filename,
                    "text": transcription["transcription"],
                    "wer": transcription.get("wer", "N/A")  # If WER is calculated
                })

        return render_template(
            'upload_audio.html', models=models, results=results, model_name=model_id, language_name=language
        )

    return render_template('upload_audio.html', models=models)


# Helper function to get language name from code
def get_language_name(code):
    languages = {
        'hi': 'Hindi',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'or': 'Oriya',
        'ta': 'Tamil',
        'te': 'Telugu',
        'en': 'English',
        'other': 'Other'
    }
    return languages.get(code, code)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('environments', exist_ok=True)
    app.run(debug=True)