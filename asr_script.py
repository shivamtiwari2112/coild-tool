import argparse
import os
import torch
import librosa
import json
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
import editdistance

# Define dataset and model paths
dataset_paths = {
    "mucs-hindi": ["/raid/home/shada/benchmark_tool/hinid_mucs_data/audio", 
                    "/raid/home/shada/benchmark_tool/hinid_mucs_data/transcription.txt"],
    "mucs-marathi": ["/raid/home/shada/benchmark_tool/marathi_mucs_data/audio", 
                      "/raid/home/shada/benchmark_tool/marathi_mucs_data/transcription.txt"],
}

model_paths = {
    "W2V-Bert-Hindi": "/raid/home/shada/benchmark_tool/w2v_bert_hindi_model/",
    "W2V-Bert-Marathi": "/raid/home/shada/benchmark_tool/w2v_bert_marathi_model",
}

def load_audio_from_dataset(dataset_name):
    """Load audio file paths and transcriptions from the dataset."""
    true_transcriptions = []
    audio_path = dataset_paths[dataset_name][0]
    label_path = dataset_paths[dataset_name][1]
    
    with open(label_path, 'r', encoding='utf-8') as label_file:
        for line in label_file:
            line = line.strip().split(' ', 1)
            file_name = line[0].strip()
            transcription = line[1].strip() if len(line) > 1 else ""
            true_transcriptions.append(transcription)
    
    audio_files = sorted([os.path.join(audio_path, file) for file in os.listdir(audio_path) if file.endswith(".wav")])
    return {'audio': audio_files[:50], 'text': true_transcriptions[:50]}

def calculate_wer(predictions, ground_truths):
    """Calculate the Word Error Rate (WER)."""
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    total_edit_distance = 0
    total_ground_truth_words = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = pred.lower().split()
        gt_tokens = gt.lower().split()
        edit_distance = editdistance.eval(pred_tokens, gt_tokens)
        total_edit_distance += edit_distance
        total_ground_truth_words += len(gt_tokens)
    
    if total_ground_truth_words == 0:
        return 0.0
    
    wer = (total_edit_distance / total_ground_truth_words) * 100
    return wer

def get_confidence_score(logits):
    """Calculate a simple confidence score based on the logits."""
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Get the maximum probability for each position
    max_probs = torch.max(probs, dim=-1)[0]
    # Return the mean of the max probabilities as a confidence score
    return float(torch.mean(max_probs).cpu().numpy())

def transcribe_audio(model_name, language, dataset_name=None, audio_file=None, reference_text=None):
    """Transcribe audio files using the selected ASR model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'IndicWav2Vec' and language == 'hi':
        model_name = 'W2V-Bert-Hindi'
    
    model = Wav2Vec2BertForCTC.from_pretrained(model_paths[model_name]).to(device)
    processor = Wav2Vec2BertProcessor.from_pretrained(model_paths[model_name], unk_token="[UNK]", pad_token="[PAD]",
                                                      word_delimiter_token="|")
    
    # Single audio file mode
    if audio_file:
        print(f'Processing file: {audio_file}')
        with torch.no_grad():
            wav, sr = librosa.load(audio_file, sr=16000)
            input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            logits = model(input_features).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            transcription = processor.decode(pred_ids)
            
            result = {
                'selected_model': model_name,
                'transcription': transcription,
                'confidence': get_confidence_score(logits)
            }
            
            # Calculate WER if reference text is provided
            if reference_text:
                wer = calculate_wer(transcription, reference_text)
                result['wer'] = f"{wer:.2f}%"
                
            return result
    
    # Dataset mode
    elif dataset_name:
        print(f'Dataset name: {dataset_name}')
        data_dict = load_audio_from_dataset(dataset_name)
        gold_data = data_dict['text']
        audio_data = data_dict['audio']
        
        predictions = []
        
        for i, audio_path in enumerate(audio_data):
            print(f'Processing file {i+1}/{len(audio_data)}: {audio_path}')
            with torch.no_grad():
                wav, sr = librosa.load(audio_path, sr=16000)
                input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.to(device)
                logits = model(input_features).logits
                pred_ids = torch.argmax(logits, dim=-1)[0]
                pred_text = processor.decode(pred_ids)
                predictions.append(pred_text)
        
        wer = calculate_wer(predictions, gold_data)
        result = {
            'selected_model': model_name,
            'selected_dataset': dataset_name,
            'predictions': predictions,
            'wer': f"{wer:.2f}%"
        }
        
        return result
    
    else:
        raise ValueError("Either dataset_name or audio_file must be provided")

def main():
    parser = argparse.ArgumentParser(description="ASR Benchmarking Script")
    parser.add_argument("--model", required=True, help="Model name (e.g., W2V-Bert-Hindi or W2V-Bert-Marathi)")
    parser.add_argument("--dataset", help="Dataset name (e.g., mucs-hindi or mucs-marathi)")
    parser.add_argument("--audio", help="Path to a single audio file for transcription")
    parser.add_argument("--reference", help="Reference text for WER calculation (optional, used with --audio)")
    parser.add_argument("--output", default="output.json", help="Output file to store results")
    
    args = parser.parse_args()
    
    if args.model not in model_paths:
        raise ValueError("Invalid model selection.")
    
    if args.dataset and args.audio:
        raise ValueError("Please specify either --dataset or --audio, not both.")
    
    if args.dataset and args.dataset not in dataset_paths:
        raise ValueError("Invalid dataset selection.")
    
    if args.dataset:
        # Dataset mode
        results = transcribe_audio(args.model, dataset_name=args.dataset)
    elif args.audio:
        # Single audio file mode
        if not os.path.exists(args.audio):
            raise ValueError(f"Audio file not found: {args.audio}")
        results = transcribe_audio(args.model, audio_file=args.audio, reference_text=args.reference)
    else:
        raise ValueError("Either --dataset or --audio must be specified.")
    
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Also print the transcription to console for convenience when processing single files
    if args.audio:
        print(f"\nTranscription: {results['transcription']}")
        if 'wer' in results:
            print(f"WER: {results['wer']}")
        print(f"Confidence: {results['confidence']:.2f}")
    
if __name__ == "__main__":
    main()