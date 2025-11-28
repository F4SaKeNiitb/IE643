"""
Batched Evaluation Script: Compare Hybrid Model vs Baseline Whisper
Uses Whisper's Native Generation Config to Fix Repetition Issues
Generates grouped bar plots for WER/CER across F0, F1, F2, F3 audio levels
"""

import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from jiwer import wer, cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import gc

# Import from your main script
from backpropnetwork import (
    HybridWav2VecWhisperModel,
    HybridModelTrainer,
    ModelConfig,
    collate_fn
)


class ModelComparator:
    def __init__(self, hybrid_checkpoint_path, device='cuda:1'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load baseline Whisper with proper configuration
        print("\nLoading baseline Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        ).to(self.device)
        self.whisper_model.eval()
        
        # Configure Whisper's generation config to prevent repetition
        self.whisper_model.generation_config.repetition_penalty = 1.5
        self.whisper_model.generation_config.no_repeat_ngram_size = 3
        self.whisper_model.generation_config.max_length = 448  # Whisper's native max length
        self.whisper_model.generation_config.num_beams = 5
        self.whisper_model.generation_config.length_penalty = 1.0
        self.whisper_model.generation_config.early_stopping = True
        
        # Critical: Set language to English and task to transcribe
        self.whisper_model.generation_config.language = "en"
        self.whisper_model.generation_config.task = "transcribe"
        
        # Force tokens for proper decoding
        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            language="en", 
            task="transcribe"
        )
        self.whisper_model.generation_config.forced_decoder_ids = forced_decoder_ids
        
        print("✓ Whisper generation config:")
        print(f"  - Language: {self.whisper_model.generation_config.language}")
        print(f"  - Task: {self.whisper_model.generation_config.task}")
        print(f"  - Repetition penalty: {self.whisper_model.generation_config.repetition_penalty}")
        print(f"  - No repeat ngram size: {self.whisper_model.generation_config.no_repeat_ngram_size}")
        print(f"  - Num beams: {self.whisper_model.generation_config.num_beams}")
        
        # Load hybrid model
        print("\nLoading hybrid model...")
        self.config = ModelConfig()
        self.trainer = HybridModelTrainer(self.config)
        self.trainer.device = self.device
        self.trainer.model = self.trainer.model.to(self.device)
        
        if os.path.exists(hybrid_checkpoint_path):
            checkpoint = torch.load(hybrid_checkpoint_path, map_location=self.device, weights_only=False)
            self.trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Hybrid model loaded from {hybrid_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {hybrid_checkpoint_path}")
        
        self.trainer.model.eval()
        
        # Optimize for inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def load_and_preprocess_audio(self, audio_path, target_sr=16000, max_length=30):
        """Load and preprocess a single audio file"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform_np = waveform.squeeze().numpy()
            
            # Normalize
            waveform_np = np.nan_to_num(waveform_np, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.abs(waveform_np).max()
            if max_val > 1e-6:
                waveform_np = waveform_np / max_val
            
            # Pad or truncate
            max_samples = max_length * target_sr
            if len(waveform_np) > max_samples:
                waveform_np = waveform_np[:max_samples]
            elif len(waveform_np) < max_samples:
                waveform_np = np.pad(waveform_np, (0, max_samples - len(waveform_np)))
            
            return waveform_np
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(max_length * target_sr)
    
    def batch_transcribe_whisper(self, audio_paths, batch_size=8):
        """Batch transcription with baseline Whisper using native config"""
        all_transcriptions = []
        
        num_batches = (len(audio_paths) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(audio_paths), batch_size), 
                    desc="Baseline Whisper", 
                    total=num_batches,
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for i in pbar:
            batch_paths = audio_paths[i:i+batch_size]
            waveforms = [self.load_and_preprocess_audio(path) for path in batch_paths]
            
            inputs = self.whisper_processor(
                waveforms, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            input_features = inputs.input_features.to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Use the model's configured generation_config
                    generated_ids = self.whisper_model.generate(
                        input_features,
                        generation_config=self.whisper_model.generation_config
                    )
            
            transcriptions = self.whisper_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            all_transcriptions.extend(transcriptions)
            
            # Update progress bar
            pbar.set_postfix({'samples': len(all_transcriptions)})
            
            # Clear memory
            del input_features, generated_ids
        
        return all_transcriptions
    
    def batch_transcribe_hybrid(self, audio_paths, batch_size=8):
        """Batch transcription with hybrid model"""
        all_transcriptions = []
        
        num_batches = (len(audio_paths) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(audio_paths), batch_size), 
                    desc="Hybrid Model    ", 
                    total=num_batches,
                    unit="batch",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for i in pbar:
            batch_paths = audio_paths[i:i+batch_size]
            
            # Prepare batch data
            wav2vec_inputs_list = []
            whisper_inputs_list = []
            
            for path in batch_paths:
                waveform_np = self.load_and_preprocess_audio(path)
                
                # Wav2Vec processing
                wav2vec_inputs = self.trainer.wav2vec_processor(
                    waveform_np, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )
                wav2vec_inputs_list.append(wav2vec_inputs.input_values.squeeze(0))
                
                # Whisper processing
                whisper_inputs = self.trainer.whisper_processor(
                    waveform_np, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                whisper_inputs_list.append(whisper_inputs.input_features.squeeze(0))
            
            # Stack into batches
            from torch.nn.utils.rnn import pad_sequence
            wav2vec_batch = pad_sequence(wav2vec_inputs_list, batch_first=True, padding_value=0)
            whisper_batch = torch.stack(whisper_inputs_list)
            
            wav2vec_batch = wav2vec_batch.to(self.device)
            whisper_batch = whisper_batch.to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.trainer.model(
                        wav2vec_input_values=wav2vec_batch,
                        whisper_input_features=whisper_batch
                    )
                
                generated_ids = outputs['generated_ids']
                transcriptions = self.trainer.whisper_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            
            all_transcriptions.extend(transcriptions)
            
            # Update progress bar
            pbar.set_postfix({'samples': len(all_transcriptions)})
            
            # Clear memory
            del wav2vec_batch, whisper_batch, outputs, generated_ids
        
        return all_transcriptions
    
    def evaluate_by_level_batched(self, csv_path, audio_root, batch_size=8, max_samples_per_level=None):
        """Batched evaluation for faster processing"""
        
        df = pd.read_csv(csv_path)
        
        results = {
            'F0': {'whisper': {'refs': [], 'preds': []}, 'hybrid': {'refs': [], 'preds': []}},
            'F1': {'whisper': {'refs': [], 'preds': []}, 'hybrid': {'refs': [], 'preds': []}},
            'F2': {'whisper': {'refs': [], 'preds': []}, 'hybrid': {'refs': [], 'preds': []}},
            'F3': {'whisper': {'refs': [], 'preds': []}, 'hybrid': {'refs': [], 'preds': []}}
        }
        
        # Create overall progress bar for levels
        level_pbar = tqdm(['F0', 'F1', 'F2', 'F3'], 
                         desc="Overall Progress", 
                         position=0,
                         bar_format='{l_bar}{bar}| Level {n_fmt}/{total_fmt}')
        
        for level in level_pbar:
            level_pbar.set_description(f"Processing Level {level}")
            
            # Collect all audio files for this level
            audio_files = []
            for idx, row in df.iterrows():
                base = os.path.splitext(os.path.basename(row['filename']))[0]
                wav_file = f"{base}_{level}.wav"
                wav_path = os.path.join(audio_root, level, wav_file)
                
                if os.path.isfile(wav_path):
                    audio_files.append((wav_path, row['text']))
            
            if max_samples_per_level:
                audio_files = audio_files[:max_samples_per_level]
            
            print(f"\n{'='*80}")
            print(f"LEVEL {level}: {len(audio_files)} audio files")
            print('='*80)
            
            audio_paths = [f[0] for f in audio_files]
            references = [f[1] for f in audio_files]
            
            # Baseline Whisper batch processing
            whisper_preds = self.batch_transcribe_whisper(audio_paths, batch_size)
            results[level]['whisper']['refs'] = references
            results[level]['whisper']['preds'] = whisper_preds
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Hybrid Model batch processing
            hybrid_preds = self.batch_transcribe_hybrid(audio_paths, batch_size)
            results[level]['hybrid']['refs'] = references
            results[level]['hybrid']['preds'] = hybrid_preds
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Calculate metrics for this level
            whisper_wer = wer(
                [r.lower().strip() for r in results[level]['whisper']['refs']],
                [p.lower().strip() for p in results[level]['whisper']['preds']]
            )
            whisper_cer = cer(
                [r.lower().strip() for r in results[level]['whisper']['refs']],
                [p.lower().strip() for p in results[level]['whisper']['preds']]
            )
            
            hybrid_wer = wer(
                [r.lower().strip() for r in results[level]['hybrid']['refs']],
                [p.lower().strip() for p in results[level]['hybrid']['preds']]
            )
            hybrid_cer = cer(
                [r.lower().strip() for r in results[level]['hybrid']['refs']],
                [p.lower().strip() for p in results[level]['hybrid']['preds']]
            )
            
            print(f"\n{level} Results:")
            print(f"  Baseline Whisper - WER: {whisper_wer:.4f} ({whisper_wer*100:.2f}%), CER: {whisper_cer:.4f} ({whisper_cer*100:.2f}%)")
            print(f"  Hybrid Model     - WER: {hybrid_wer:.4f} ({hybrid_wer*100:.2f}%), CER: {hybrid_cer:.4f} ({hybrid_cer*100:.2f}%)")
            
            improvement_wer = ((whisper_wer - hybrid_wer) / whisper_wer * 100) if whisper_wer > 0 else 0
            improvement_cer = ((whisper_cer - hybrid_cer) / whisper_cer * 100) if whisper_cer > 0 else 0
            
            print(f"  Improvement      - WER: {improvement_wer:+.2f}%, CER: {improvement_cer:+.2f}%")
            
            # Show sample predictions
            print(f"\nSample Predictions from {level}:")
            for j in range(min(3, len(references))):
                ref = references[j]
                w_pred = whisper_preds[j]
                h_pred = hybrid_preds[j]
                
                print(f"\n  Sample {j+1}:")
                print(f"    Ref:     {ref}")
                print(f"    Whisper: {w_pred}")
                print(f"    Hybrid:  {h_pred}")
        
        print()
        return results
    
    def calculate_metrics_summary(self, results):
        """Calculate WER and CER for each level and model"""
        summary = {
            'levels': ['F0', 'F1', 'F2', 'F3'],
            'whisper_wer': [],
            'hybrid_wer': [],
            'whisper_cer': [],
            'hybrid_cer': []
        }
        
        for level in summary['levels']:
            # Whisper metrics
            whisper_wer_val = wer(
                [r.lower().strip() for r in results[level]['whisper']['refs']],
                [p.lower().strip() for p in results[level]['whisper']['preds']]
            )
            whisper_cer_val = cer(
                [r.lower().strip() for r in results[level]['whisper']['refs']],
                [p.lower().strip() for p in results[level]['whisper']['preds']]
            )
            
            # Hybrid metrics
            hybrid_wer_val = wer(
                [r.lower().strip() for r in results[level]['hybrid']['refs']],
                [p.lower().strip() for p in results[level]['hybrid']['preds']]
            )
            hybrid_cer_val = cer(
                [r.lower().strip() for r in results[level]['hybrid']['refs']],
                [p.lower().strip() for p in results[level]['hybrid']['preds']]
            )
            
            summary['whisper_wer'].append(whisper_wer_val)
            summary['hybrid_wer'].append(hybrid_wer_val)
            summary['whisper_cer'].append(whisper_cer_val)
            summary['hybrid_cer'].append(hybrid_cer_val)
        
        return summary
    
    def plot_comparison(self, summary, output_dir='./evaluation_results'):
        """Create grouped bar plots for WER and CER comparison"""
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        x = np.arange(len(summary['levels']))
        width = 0.35
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # WER Plot
        bars1 = ax1.bar(x - width/2, [w*100 for w in summary['whisper_wer']], 
                        width, label='Baseline Whisper', color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, [w*100 for w in summary['hybrid_wer']], 
                        width, label='Hybrid Model', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Audio Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('WER Comparison: Baseline vs Hybrid Model', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary['levels'])
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(max(summary['whisper_wer']), max(summary['hybrid_wer'])) * 110)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # CER Plot
        bars3 = ax2.bar(x - width/2, [c*100 for c in summary['whisper_cer']], 
                        width, label='Baseline Whisper', color='#3498db', alpha=0.8, edgecolor='black')
        bars4 = ax2.bar(x + width/2, [c*100 for c in summary['hybrid_cer']], 
                        width, label='Hybrid Model', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Audio Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Character Error Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('CER Comparison: Baseline vs Hybrid Model', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(summary['levels'])
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(max(summary['whisper_cer']), max(summary['hybrid_cer'])) * 110)
        
        # Add value labels on bars
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / 'wer_cer_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {plot_path}")
        plt.show()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Level': summary['levels'],
            'Whisper_WER': [f"{w*100:.2f}%" for w in summary['whisper_wer']],
            'Hybrid_WER': [f"{w*100:.2f}%" for w in summary['hybrid_wer']],
            'WER_Improvement': [f"{((w-h)/w*100):+.2f}%" if w > 0 else "N/A" 
                               for w, h in zip(summary['whisper_wer'], summary['hybrid_wer'])],
            'Whisper_CER': [f"{c*100:.2f}%" for c in summary['whisper_cer']],
            'Hybrid_CER': [f"{c*100:.2f}%" for c in summary['hybrid_cer']],
            'CER_Improvement': [f"{((c-h)/c*100):+.2f}%" if c > 0 else "N/A"
                               for c, h in zip(summary['whisper_cer'], summary['hybrid_cer'])]
        })
        
        csv_path = Path(output_dir) / 'metrics_comparison.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved to {csv_path}")
        
        return metrics_df


def main():
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    HYBRID_CHECKPOINT = './checkpoints_tiny/checkpoint_latest_epoch_9.pt'
    CSV_PATH = "Feebles_test/Transcriptions_test.csv"
    AUDIO_ROOT = "Feebles_test"
    MAX_SAMPLES_PER_LEVEL = 1000  # Set to None to evaluate all samples
    BATCH_SIZE = 16
    OUTPUT_DIR = 'evaluation_results_final_tiny'
    DEVICE = 'cuda:1'
    
    print("="*80)
    print("FIXED MODEL COMPARISON: HYBRID vs BASELINE WHISPER")
    print("Using Whisper's Native Generation Config")
    print("="*80)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Samples per level: {MAX_SAMPLES_PER_LEVEL if MAX_SAMPLES_PER_LEVEL else 'All'}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Initialize comparator
    comparator = ModelComparator(
        hybrid_checkpoint_path=HYBRID_CHECKPOINT,
        device=DEVICE
    )
    
    # Evaluate both models on each level with batching
    results = comparator.evaluate_by_level_batched(
        csv_path=CSV_PATH,
        audio_root=AUDIO_ROOT,
        batch_size=BATCH_SIZE,
        max_samples_per_level=MAX_SAMPLES_PER_LEVEL
    )
    
    # Calculate summary metrics
    print("\n" + "="*80)
    print("CALCULATING SUMMARY METRICS")
    print("="*80)
    summary = comparator.calculate_metrics_summary(results)
    
    # Create comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    metrics_df = comparator.plot_comparison(summary, output_dir=OUTPUT_DIR)
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL METRICS SUMMARY")
    print("="*80)
    print(metrics_df.to_string(index=False))
    
    # Save detailed results
    print("\nSaving detailed predictions...")
    detailed_results = []
    for level in ['F0', 'F1', 'F2', 'F3']:
        for i in range(len(results[level]['whisper']['refs'])):
            detailed_results.append({
                'Level': level,
                'Reference': results[level]['whisper']['refs'][i],
                'Whisper_Prediction': results[level]['whisper']['preds'][i],
                'Hybrid_Prediction': results[level]['hybrid']['preds'][i]
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = Path(OUTPUT_DIR) / 'detailed_predictions.csv'
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed predictions saved to {detailed_path}")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    avg_whisper_wer = np.mean(summary['whisper_wer']) * 100
    avg_hybrid_wer = np.mean(summary['hybrid_wer']) * 100
    avg_whisper_cer = np.mean(summary['whisper_cer']) * 100
    avg_hybrid_cer = np.mean(summary['hybrid_cer']) * 100
    
    overall_wer_improvement = ((avg_whisper_wer - avg_hybrid_wer) / avg_whisper_wer * 100)
    overall_cer_improvement = ((avg_whisper_cer - avg_hybrid_cer) / avg_whisper_cer * 100)
    
    print(f"Average WER:")
    print(f"  Baseline Whisper: {avg_whisper_wer:.2f}%")
    print(f"  Hybrid Model:     {avg_hybrid_wer:.2f}%")
    print(f"  Improvement:      {overall_wer_improvement:+.2f}%")
    
    print(f"\nAverage CER:")
    print(f"  Baseline Whisper: {avg_whisper_cer:.2f}%")
    print(f"  Hybrid Model:     {avg_hybrid_cer:.2f}%")
    print(f"  Improvement:      {overall_cer_improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("  - wer_cer_comparison.png (visualization)")
    print("  - metrics_comparison.csv (summary metrics)")
    print("  - detailed_predictions.csv (all predictions)")


if __name__ == "__main__":
    main()
