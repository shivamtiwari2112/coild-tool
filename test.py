import pandas as pd
import json
import os

def create_asr_data_json():
    # Your provided DataFrame data
    base_df = pd.DataFrame({
        "Dataset": ["mucs", "mucs", "mucs", "mucs", "mucs", "openslr", "kathbath", "shrutilipi"],
        "Language": ["hi", "mr", "gu", "or", "ta", "te", "hi", "hi"],
        "IndicWav2Vec": [9.5, 11.7, 14.3, 20.6, 19.5, 15.1, 8.9, 9.2],
        "Whisper": [7.2, 8.5, 10.2, 15.3, 14.2, 11.0, 6.5, 7.1],
        "XLS-R": [8.1, 9.8, 12.1, 17.5, 16.8, 12.5, 7.3, 8.0],
        "HuBERT": [8.9, 10.5, 13.0, 18.2, 17.5, 13.2, 7.8, 8.5],
        "mSLAM": [6.8, 8.1, 9.8, 14.8, 13.7, 10.5, 6.0, 6.5]
    })
    
    # Model metadata
    model_metadata = {
        "IndicWav2Vec": {
            "Year": 2020,
            "Extra_Training": "✓",
            "Papers": "https://arxiv.org/abs/2105.03595",
            "Code": "https://github.com/AI4Bharat/IndicWav2Vec",
        },
        "Whisper": {
            "Year": 2022,
            "Extra_Training": "✓",
            "Papers": "https://arxiv.org/abs/2212.04356",
            "Code": "https://github.com/openai/whisper",
        },
        "XLS-R": {
            "Year": 2020,
            "Extra_Training": "✓",
            "Papers": "https://arxiv.org/abs/2111.09296",
            "Code": "https://github.com/pytorch/fairseq",
        },
        "HuBERT": {
            "Year": 2021,
            "Extra_Training": "✗",
            "Papers": "https://arxiv.org/abs/2106.07447",
            "Code": "https://github.com/facebookresearch/fairseq",
        },
        "mSLAM": {
            "Year": 2022,
            "Extra_Training": "✓",
            "Papers": "https://arxiv.org/abs/2305.10599",
            "Code": "https://github.com/microsoft/SpeechT5"
        }
    }
    
    # Create a structure compatible with your app
    output_data = {}
    
    # Get unique datasets and languages
    datasets = base_df["Dataset"].unique()
    
    for dataset in datasets:
        output_data[dataset] = {}
        
        # Filter dataset rows
        dataset_df = base_df[base_df["Dataset"] == dataset]
        
        for _, row in dataset_df.iterrows():
            language = row["Language"]
            
            if language not in output_data[dataset]:
                output_data[dataset][language] = []
            
            # Add each model's data for this dataset and language
            for model_name in ["IndicWav2Vec", "Whisper", "XLS-R", "HuBERT", "mSLAM"]:
                if not pd.isna(row[model_name]):  # Check if WER value exists
                    # Convert "✓" to boolean True and "✗" to boolean False
                    extra_training = True if model_metadata[model_name]["Extra_Training"] == "✓" else False
                    
                    model_data = {
                        "model": model_name,
                        "test_wer": row[model_name],
                        "year": model_metadata[model_name]["Year"],
                        "extra_training_data": extra_training,
                        "paper": model_metadata[model_name]["Papers"],
                        "code": model_metadata[model_name]["Code"]
                    }
                    
                    output_data[dataset][language].append(model_data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Write the JSON file
    with open('data/asr_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("ASR data JSON created successfully at data/asr_data.json")
    return output_data

if __name__ == "__main__":
    create_asr_data_json()