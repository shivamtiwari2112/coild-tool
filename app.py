# # app.py
# from flask import Flask, render_template, request, jsonify
# import json
# import os
# import pandas as pd

# app = Flask(__name__)

# # Load data from a JSON file
# def load_data():
#     if os.path.exists('data/asr_data.json'):
#         with open('data/asr_data.json', 'r') as f:
#             return json.load(f)
#     # return create_default_data()

# # Create default data if none exists
# # def create_default_data():
#     # try:
#     #     # Create a pandas DataFrame with the data
#     #     base_df = pd.DataFrame({
#     #         "Dataset": ["mucs", "mucs", "mucs", "mucs", "mucs", "openslr", "kathbath", "shrutilipi"],
#     #         "Language": ["hi", "mr", "gu", "or", "ta", "te", "hi", "hi"],
#     #         "IndicWav2Vec": [9.5, 11.7, 14.3, 20.6, 19.5, 15.1, 8.9, 9.2],
#     #         "Whisper": [7.2, 8.5, 10.2, 15.3, 14.2, 11.0, 6.5, 7.1],
#     #         "XLS-R": [8.1, 9.8, 12.1, 17.5, 16.8, 12.5, 7.3, 8.0],
#     #         "HuBERT": [8.9, 10.5, 13.0, 18.2, 17.5, 13.2, 7.8, 8.5],
#     #         "mSLAM": [6.8, 8.1, 9.8, 14.8, 13.7, 10.5, 6.0, 6.5]
#     #     })
        
#     #     # Model metadata
#     #     model_metadata = {
#     #         "IndicWav2Vec": {
#     #             "Year": 2020,
#     #             "Extra_Training": True,
#     #             "Papers": "https://arxiv.org/abs/2105.03595",
#     #             "Code": "https://github.com/AI4Bharat/IndicWav2Vec",
#     #         },
#     #         "Whisper": {
#     #             "Year": 2022,
#     #             "Extra_Training": True,
#     #             "Papers": "https://arxiv.org/abs/2212.04356",
#     #             "Code": "https://github.com/openai/whisper",
#     #         },
#     #         "XLS-R": {
#     #             "Year": 2020,
#     #             "Extra_Training": True,
#     #             "Papers": "https://arxiv.org/abs/2111.09296",
#     #             "Code": "https://github.com/pytorch/fairseq",
#     #         },
#     #         "HuBERT": {
#     #             "Year": 2021,
#     #             "Extra_Training": False,
#     #             "Papers": "https://arxiv.org/abs/2106.07447",
#     #             "Code": "https://github.com/facebookresearch/fairseq",
#     #         },
#     #         "mSLAM": {
#     #             "Year": 2022,
#     #             "Extra_Training": True,
#     #             "Papers": "https://arxiv.org/abs/2305.10599",
#     #             "Code": "https://github.com/microsoft/SpeechT5"
#     #         }
#     #     }
        
#     #     # Create output structure
#     #     output_data = {}
        
#     #     # Get unique datasets and languages
#     #     datasets = base_df["Dataset"].unique()
        
#     #     for dataset in datasets:
#     #         output_data[dataset] = {}
            
#     #         # Filter dataset rows
#     #         dataset_df = base_df[base_df["Dataset"] == dataset]
            
#     #         for _, row in dataset_df.iterrows():
#     #             language = row["Language"]
                
#     #             if language not in output_data[dataset]:
#     #                 output_data[dataset][language] = []
                
#     #             # Add each model's data for this dataset and language
#     #             for model_name in ["IndicWav2Vec", "Whisper", "XLS-R", "HuBERT", "mSLAM"]:
#     #                 if not pd.isna(row[model_name]):  # Check if WER value exists
#     #                     model_data = {
#     #                         "model": model_name,
#     #                         "test_wer": row[model_name],
#     #                         "year": model_metadata[model_name]["Year"],
#     #                         "extra_training_data": model_metadata[model_name]["Extra_Training"],
#     #                         "paper": model_metadata[model_name]["Papers"],
#     #                         "code": model_metadata[model_name]["Code"]
#     #                     }
                        
#     #                     output_data[dataset][language].append(model_data)
        
#     #     # Create directories
#     #     os.makedirs('data', exist_ok=True)
        
#     #     # Save the data
#     #     with open('data/asr_data.json', 'w') as f:
#     #         json.dump(output_data, f, indent=2)
        
#     #     return output_data
    
#     # except Exception as e:
#     #     print(f"Error creating default data: {e}")
        
#     #     # Return a minimal sample structure as fallback
#     #     return {
#     #         "mucs": {
#     #             "hi": [
#     #                 {"model": "mSLAM", "test_wer": 6.8, "year": 2022, "extra_training_data": True, "paper": "https://arxiv.org/abs/2305.10599", "code": "https://github.com/microsoft/SpeechT5"},
#     #                 {"model": "Whisper", "test_wer": 7.2, "year": 2022, "extra_training_data": True, "paper": "https://arxiv.org/abs/2212.04356", "code": "https://github.com/openai/whisper"}
#     #             ]
#     #         }
#     #     }

# @app.route('/')
# def index():
#     data = load_data()
    
#     # Get all available datasets
#     datasets = list(data.keys())
    
#     # Default to first dataset if available, or use query parameter
#     selected_dataset = request.args.get('dataset', datasets[0] if datasets else None)
    
#     # Initialize variables with defaults
#     languages = []
#     selected_language = None
#     models = []
    
#     if selected_dataset and selected_dataset in data:
#         # Get all languages for the selected dataset
#         languages = list(data[selected_dataset].keys())
        
#         # Default to first language if available, or use query parameter
#         selected_language = request.args.get('language', languages[0] if languages else None)
        
#         if selected_language and selected_language in data[selected_dataset]:
#             # Get models for the selected dataset and language
#             models_data = data[selected_dataset][selected_language]
            
#             # Ensure all models have the required fields
#             for model in models_data:
#                 if 'test_wer' not in model:
#                     # If 'test_wer' is missing but 'wer' exists, use that
#                     if 'wer' in model:
#                         model['test_wer'] = model['wer']
#                     else:
#                         # Otherwise set a default value
#                         model['test_wer'] = 100.0  # Default high WER
            
#             # Sort models by WER (ascending)
#             models = sorted(models_data, key=lambda x: x.get('test_wer', 100.0))
            
#             # Add rank to each model
#             for i, model in enumerate(models):
#                 model['rank'] = i + 1
    
#     return render_template('index.html', 
#                           datasets=datasets,
#                           languages=languages,
#                           selected_dataset=selected_dataset,
#                           selected_language=selected_language,
#                           models=models)

# @app.route('/api/data')
# def get_data():
#     data = load_data()
#     dataset = request.args.get('dataset')
#     language = request.args.get('language')
    
#     if dataset in data and language in data[dataset]:
#         return jsonify(data[dataset][language])
#     return jsonify([])

# @app.route('/api/datasets')
# def get_datasets():
#     data = load_data()
#     return jsonify(list(data.keys()))

# @app.route('/api/languages')
# def get_languages():
#     data = load_data()
#     dataset = request.args.get('dataset')
    
#     if dataset in data:
#         return jsonify(list(data[dataset].keys()))
#     return jsonify([])

# if __name__ == '__main__':
    
#     app.run(debug=True)



# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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

@app.route('/upload', methods=['GET', 'POST'])
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
            return render_template('upload.html', datasets=datasets, message="Model name is required", message_type="danger")
        
        if not dataset or not language:
            flash('Dataset and language are required', 'danger')
            return render_template('upload.html', datasets=datasets, message="Dataset and language are required", message_type="danger")
        
        # Check required files
        if 'model_weights' not in request.files or not request.files['model_weights'].filename:
            return render_template('upload.html', datasets=datasets, message="Model weights file is required", message_type="danger")
        
        if 'requirements_file' not in request.files or not request.files['requirements_file'].filename:
            return render_template('upload.html', datasets=datasets, message="Requirements file is required", message_type="danger")
        
        if 'inference_file' not in request.files or not request.files['inference_file'].filename:
            return render_template('upload.html', datasets=datasets, message="Inference file is required", message_type="danger")
        
        # Get files
        model_weights = request.files['model_weights']
        requirements_file = request.files['requirements_file']
        inference_file = request.files['inference_file']
        
        # Validate file types
        if not allowed_file(model_weights.filename, 'model_weights'):
            return render_template('upload.html', datasets=datasets, message="Invalid model weights file format", message_type="danger")
        
        if not allowed_file(requirements_file.filename, 'requirements'):
            return render_template('upload.html', datasets=datasets, message="Invalid requirements file format", message_type="danger")
        
        if not allowed_file(inference_file.filename, 'inference'):
            return render_template('upload.html', datasets=datasets, message="Invalid inference file format", message_type="danger")
        
        # Check file names
        if requirements_file.filename != 'requirements.txt':
            return render_template('upload.html', datasets=datasets, message="Requirements file must be named 'requirements.txt'", message_type="danger")
        
        if inference_file.filename != 'model_inference.py':
            return render_template('upload.html', datasets=datasets, message="Inference file must be named 'model_inference.py'", message_type="danger")
        
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
    
    return render_template('upload.html', datasets=datasets)

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

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True)