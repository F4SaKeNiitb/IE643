import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchaudio
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WhisperForConditionalGeneration, WhisperProcessor
)
import numpy as np
from pathlib import Path
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import gc

@dataclass
class ModelConfig:
    wav2vec_model = "facebook/wav2vec2-large-960h"
    whisper_model = "openai/whisper-tiny"
    max_audio_length = 30
    sample_rate = 16000
    fusion_hidden_dim = 512
    fusion_dropout = 0.1
    batch_size = 8  
    learning_rate = 5e-6
    num_epochs = 10
    gradient_accumulation_steps = 2  # To maintain effective batch_size=16
    output_dir = "./hybrid_model_output_tiny"
    checkpoint_dir = "./checkpoints_tiny"
    max_grad_norm = 1.0
    use_wav2vec = False
    fusion_weight = 0.3
    min_generation_length = 10
    length_penalty = 2.5
    warmup_steps = 100

class AudioDataset(Dataset):
    def __init__(self, audio_paths, transcriptions, wav2vec_processor,
                 whisper_processor, max_length=30, sample_rate=16000):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.wav2vec_processor = wav2vec_processor
        self.whisper_processor = whisper_processor
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = torch.zeros(1, self.max_length * self.sample_rate)
            sr = self.sample_rate

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze().numpy()
        
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_val = np.abs(waveform).max()
        if max_val > 1e-6:
            waveform = waveform / max_val
        else:
            waveform = np.zeros_like(waveform)

        max_samples = self.max_length * self.sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        elif len(waveform) < max_samples:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)))

        wav2vec_inputs = self.wav2vec_processor(
            waveform, sampling_rate=self.sample_rate, 
            return_tensors="pt", padding=True
        )

        whisper_inputs = self.whisper_processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        labels = self.whisper_processor.tokenizer(
            self.transcriptions[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).input_ids.squeeze(0)

        return {
            'wav2vec_input_values': wav2vec_inputs.input_values.squeeze(0),
            'whisper_input_features': whisper_inputs.input_features.squeeze(0),
            'labels': labels,
            'transcription': self.transcriptions[idx]
        }

class AttentionFusionLayer(nn.Module):
    """Cross-attention fusion between Wav2Vec2 and Whisper features"""
    def __init__(self, wav2vec_dim=1024, whisper_dim=384, fusion_dim=512, dropout=0.1, num_heads=8):
        super().__init__()
        
        self.wav2vec_proj = nn.Linear(wav2vec_dim, fusion_dim)
        self.whisper_proj = nn.Linear(whisper_dim, fusion_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.output_proj = nn.Linear(fusion_dim, whisper_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, wav2vec_features, whisper_features, fusion_weight=0.3):
        wav2vec_proj = self.wav2vec_proj(wav2vec_features)
        whisper_proj = self.whisper_proj(whisper_features)
        
        if wav2vec_proj.size(1) != whisper_proj.size(1):
            wav2vec_proj = F.interpolate(
                wav2vec_proj.transpose(1, 2), 
                size=whisper_proj.size(1),
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        attn_output, _ = self.cross_attention(
            whisper_proj,
            wav2vec_proj,
            wav2vec_proj
        )
        
        fused = self.norm1(whisper_proj + attn_output)
        fused = self.norm2(fused + self.ffn(fused))
        output = self.output_proj(fused)
        output = (1 - fusion_weight) * whisper_features + fusion_weight * output
        
        return output

class HybridWav2VecWhisperModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wav2vec = Wav2Vec2Model.from_pretrained(config.wav2vec_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            config.whisper_model
        )

        self._freeze_base_models()

        wav2vec_dim = self.wav2vec.config.hidden_size
        whisper_dim = self.whisper.config.d_model

        self.fusion = AttentionFusionLayer(
            wav2vec_dim=wav2vec_dim, 
            whisper_dim=whisper_dim,
            fusion_dim=config.fusion_hidden_dim,
            dropout=config.fusion_dropout
        )

    def _freeze_base_models(self):
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        for param in self.whisper.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze decoder for fine-tuning"""
        for param in self.whisper.model.decoder.parameters():
            param.requires_grad = True

    def forward(self, wav2vec_input_values, whisper_input_features, 
                labels=None, return_dict=True):
    
        whisper_input_features = torch.nan_to_num(whisper_input_features, nan=0.0)
        
        # Use no_grad for encoder to save memory
        with torch.no_grad():
            whisper_encoder_outputs = self.whisper.model.encoder(whisper_input_features)
        whisper_features = whisper_encoder_outputs.last_hidden_state
        
        use_fusion = self.config.use_wav2vec and self.training
        
        if use_fusion:
            try:
                wav2vec_input_values = torch.nan_to_num(wav2vec_input_values, nan=0.0)
                with torch.no_grad():  # Wav2Vec always frozen
                    with torch.cuda.amp.autocast(enabled=False):
                        wav2vec_input_fp32 = wav2vec_input_values.float()
                        wav2vec_outputs = self.wav2vec(wav2vec_input_fp32)
                        wav2vec_features = wav2vec_outputs.last_hidden_state
                
                if torch.isnan(wav2vec_features).any() or torch.isinf(wav2vec_features).any():
                    use_fusion = False
                else:
                    wav2vec_features = torch.nan_to_num(wav2vec_features, nan=0.0)
                    decoder_inputs = self.fusion(
                        wav2vec_features, whisper_features, 
                        fusion_weight=self.config.fusion_weight
                    )
            except Exception as e:
                print(f"Fusion error: {e}, using Whisper-only")
                use_fusion = False
        
        if not use_fusion:
            decoder_inputs = whisper_features
        
        decoder_inputs = torch.nan_to_num(decoder_inputs, nan=0.0)
        decoder_inputs = F.layer_norm(decoder_inputs, decoder_inputs.shape[-1:])
        
        modified_encoder_outputs = type(whisper_encoder_outputs)(
            last_hidden_state=decoder_inputs
        )
    
        if labels is not None:
            decoder_outputs = self.whisper(
                encoder_outputs=modified_encoder_outputs,
                labels=labels, 
                return_dict=True
            )
    
            loss = decoder_outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                total_loss = torch.tensor(5.0, device=loss.device, requires_grad=True)
            else:
                total_loss = torch.clamp(loss, 0.0, 20.0)
    
            if total_loss.dim() > 0:
                total_loss = total_loss.mean()
    
            return {
                'loss': total_loss,
                'ce_loss': loss,
                'logits': decoder_outputs.logits
            }
        else:
            generated_ids = self.whisper.generate(
                encoder_outputs=modified_encoder_outputs,
                max_length=448,
                min_length=self.config.min_generation_length,
                num_beams=5,
                length_penalty=self.config.length_penalty,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                forced_decoder_ids=None,
            )
            return {'generated_ids': generated_ids}

class HybridModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            config.wav2vec_model
        )
        self.whisper_processor = WhisperProcessor.from_pretrained(
            config.whisper_model
        )

        self.model = HybridWav2VecWhisperModel(config)
        self.scaler = GradScaler(init_scale=256.0)  # Reduced from 512
        
        self.optimizer = None
        self.scheduler = None
        self.current_phase = 1
        self.global_step = 0

        Path(config.output_dir).mkdir(exist_ok=True, parents=True)
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)

    def prepare_dataset(self, audio_paths, transcriptions):
        dataset = AudioDataset(
            audio_paths=audio_paths, transcriptions=transcriptions,
            wav2vec_processor=self.wav2vec_processor,
            whisper_processor=self.whisper_processor,
            max_length=self.config.max_audio_length,
            sample_rate=self.config.sample_rate
        )
        
        pad_token_id = self.whisper_processor.tokenizer.pad_token_id
        
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=False,  # Changed from True
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )

    def generate_sample_predictions(self, batch, num_samples=3):
        """Generate and display sample predictions"""
        self.model.eval()
        
        with torch.no_grad():
            num_samples = min(num_samples, len(batch['transcription']))
            wav2vec_inputs = batch['wav2vec_input_values'][:num_samples].to(self.device)
            whisper_inputs = batch['whisper_input_features'][:num_samples].to(self.device)
            references = batch['transcription'][:num_samples]
            
            with autocast('cuda'):
                outputs = self.model(
                    wav2vec_input_values=wav2vec_inputs,
                    whisper_input_features=whisper_inputs
                )
            
            generated_ids = outputs['generated_ids']
            predictions = self.whisper_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            print(f"\n{'='*80}")
            print("SAMPLE PREDICTIONS")
            print('='*80)
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_norm = pred.lower().strip().replace("'", "'")
                ref_norm = ref.lower().strip().replace("'", "'")
                match = "✓" if pred_norm == ref_norm else "✗"
                
                print(f"  Sample {i+1} {match}")
                print(f"    Ref:  {ref}")
                print(f"    Pred: {pred}")
                print()
            print('='*80 + "\n")
        
        self.model.train()

    def get_lr_scale(self, step):
        """Get learning rate scale factor with warmup"""
        warmup_steps = self.config.warmup_steps
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    def train_epoch(self, dataloader, epoch):
        print(f"\nMemory before epoch: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            wav2vec_inputs = batch['wav2vec_input_values'].to(self.device)
            whisper_inputs = batch['whisper_input_features'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast('cuda', enabled=True, dtype=torch.float16):
                outputs = self.model(
                    wav2vec_input_values=wav2vec_inputs,
                    whisper_input_features=whisper_inputs,
                    labels=labels
                )
                loss = outputs['loss']
                
                if loss.dim() > 0:
                    loss = loss.mean()
                
                ce_loss_tensor = outputs['ce_loss']
                if ce_loss_tensor.dim() > 0:
                    ce_loss = ce_loss_tensor.mean().item()
                else:
                    ce_loss = ce_loss_tensor.item()
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    print(f"WARNING: Invalid loss {loss.item()}, skipping batch")
                    continue
                
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            
            # Clear intermediate tensors
            del wav2vec_inputs, whisper_inputs, labels, outputs

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.max_grad_norm
                )
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"WARNING: Invalid gradient norm, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_ce_loss += ce_loss
            num_batches += 1

            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                      f"CE: {ce_loss:.4f} | LR: {current_lr:.2e} | "
                      f"Mem: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB")
                
            if batch_idx in [1000] or (batch_idx > 0 and batch_idx % 1000 == 0):
                self.generate_sample_predictions(batch, num_samples=2)

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce = total_ce_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce
        }

    # Replace the fine_tune method signature (around line 377) from:
# def fine_tune(self, train_audio_paths, train_transcriptions, 
#               val_audio_paths=None, val_transcriptions=None):

# To:
    def fine_tune(self, train_audio_paths, train_transcriptions, 
                val_audio_paths=None, val_transcriptions=None, start_epoch=0):
        
        print("="*80)
        print("PHASED TRAINING APPROACH")
        print("="*80)
        print("Phase 1 (Epochs 1-3): Whisper-only baseline")
        print("Phase 2 (Epochs 4-6): Enable fusion with attention")
        print("Phase 3 (Epochs 7-10): Full fusion fine-tuning")
        print("="*80 + "\n")
        
        self.model.unfreeze_decoder()
        
        fusion_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fusion' in name:
                    fusion_params.append(param)
                else:
                    decoder_params.append(param)
        
        print(f"Fusion parameters: {sum(p.numel() for p in fusion_params):,}")
        print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}\n")
        
        # Only create new optimizer if starting from scratch
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW([
                {'params': fusion_params, 'lr': self.config.learning_rate},
                {'params': decoder_params, 'lr': self.config.learning_rate}
            ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        train_loader = self.prepare_dataset(train_audio_paths, train_transcriptions)
        
        # Only create new scheduler if starting from scratch
        if self.scheduler is None:
            from torch.optim.lr_scheduler import OneCycleLR
            total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )

        best_val_wer = float('inf')
        
        # Use start_epoch in the range
        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            
            if epoch < 3:
                phase = 1
                self.config.use_wav2vec = False
                phase_name = "Whisper-only baseline"
            elif epoch < 6:
                phase = 2
                self.config.use_wav2vec = True
                self.config.fusion_weight = 0.2
                phase_name = "Light fusion with attention"
            else:
                phase = 3
                self.config.use_wav2vec = True
                self.config.fusion_weight = 0.3
                phase_name = "Full fusion fine-tuning"
                
                if epoch == 6:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
            
            print(f"Phase {phase}: {phase_name}")
            print('='*80)
            
            metrics = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Total Loss: {metrics['loss']:.4f}")
            print(f"  CE Loss: {metrics['ce_loss']:.4f}")
            
            if (epoch + 1) % 2 == 0 and val_audio_paths and val_transcriptions:
                print(f"\n{'='*80}")
                print("VALIDATION")
                print('='*80)
                val_metrics = self.evaluate(val_audio_paths, val_transcriptions)
                print(f"Validation WER: {val_metrics['wer']:.4f} ({val_metrics['wer']*100:.2f}%)")
                print(f"Validation CER: {val_metrics['cer']:.4f} ({val_metrics['cer']*100:.2f}%)")
                
                print("\nSample Predictions:")
                for i in range(min(5, len(val_metrics['predictions']))):
                    ref = val_metrics['references'][i]
                    pred = val_metrics['predictions'][i]
                    
                    ref_norm = ref.lower().strip().replace("'", "'")
                    pred_norm = pred.lower().strip().replace("'", "'")
                    match = "✓" if ref_norm == pred_norm else "✗"
                    
                    print(f"  {i+1}. {match}")
                    print(f"     Ref:  {ref}")
                    print(f"     Pred: {pred}\n")
                
                if val_metrics['wer'] < best_val_wer:
                    best_val_wer = val_metrics['wer']
                    self.save_checkpoint(epoch, stage="best")
                    print(f"✓ New best model saved (WER: {best_val_wer:.4f})")
            
            self.save_checkpoint(epoch, stage="latest")
            
            # Clear cache after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)

    def save_checkpoint(self, epoch, stage=""):
        checkpoint_path = Path(self.config.checkpoint_dir) / \
                         f"checkpoint_{stage}_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']

    def evaluate(self, audio_paths, transcriptions):
        self.model.eval()
        test_loader = self.prepare_dataset(audio_paths, transcriptions)

        predictions = []
        references = []

        with torch.no_grad():
            for batch in test_loader:
                wav2vec_inputs = batch['wav2vec_input_values'].to(self.device)
                whisper_inputs = batch['whisper_input_features'].to(self.device)

                with autocast('cuda'):
                    outputs = self.model(
                        wav2vec_input_values=wav2vec_inputs,
                        whisper_input_features=whisper_inputs
                    )

                generated_ids = outputs['generated_ids']
                decoded = self.whisper_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                predictions.extend(decoded)
                references.extend(batch['transcription'])
                
                # Clear memory
                del wav2vec_inputs, whisper_inputs, outputs
                if len(predictions) % 50 == 0:
                    torch.cuda.empty_cache()

        try:
            from jiwer import wer, cer
            
            refs_norm = [r.lower().strip() for r in references]
            preds_norm = [p.lower().strip() for p in predictions]
            
            word_error_rate = wer(refs_norm, preds_norm)
            char_error_rate = cer(refs_norm, preds_norm)
        except Exception as e:
            print(f"Error calculating WER/CER: {e}")
            word_error_rate = 0.0
            char_error_rate = 0.0

        return {
            'wer': word_error_rate, 
            'cer': char_error_rate,
            'predictions': predictions, 
            'references': references
        }

def collate_fn(batch, pad_token_id):
    wav2vec_vals = [item['wav2vec_input_values'] for item in batch]
    whisper_feats = [item['whisper_input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    texts = [item['transcription'] for item in batch]
    
    wav2vec_batch = pad_sequence(wav2vec_vals, batch_first=True, padding_value=0)
    whisper_batch = torch.stack(whisper_feats)
    labels_batch = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    
    return {
        'wav2vec_input_values': wav2vec_batch,
        'whisper_input_features': whisper_batch,
        'labels': labels_batch,
        'transcription': texts
    }

def load_and_validate_audio(args):
    """Load and validate a single audio file"""
    wav_path, text, sample_rate, max_length = args
    
    try:
        if not os.path.isfile(wav_path):
            return ('missing', None)
            
        waveform, sr = torchaudio.load(wav_path)
        
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            return ('nan_inf', None)
        
        max_amp = waveform.abs().max()
        if max_amp < 1e-6:
            return ('silent', None)
        
        return ('success', (wav_path, text))
    
    except Exception as e:
        return ('error', str(e))


def load_dataset_parallel(csv_path, audio_root, max_samples=None, num_workers=None):
    """Load dataset using multiple CPU threads"""
    
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 20)
    
    print(f"Using {num_workers} CPU threads for dataset loading...")
    
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df[:max_samples]
    
    tasks = []
    for idx, row in df.iterrows():
        base = os.path.splitext(os.path.basename(row['filename']))[0]
    
        for level in ['F0', 'F1', 'F2', 'F3']:
            wav_file = f"{base}_{level}.wav"
            wav_path = os.path.join(audio_root, level, wav_file)
            
            if os.path.isfile(wav_path):
                tasks.append((wav_path, row['text'], 16000, 30))
    
    print(f"Found {len(tasks)} audio files to validate")
    
    audio_paths = []
    transcriptions = []
    
    failure_counts = {
        'missing': 0,
        'nan_inf': 0,
        'silent': 0,
        'error': 0
    }
    error_samples = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_and_validate_audio, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating audio"):
            status, result = future.result()
            
            if status == 'success':
                wav_path, text = result
                audio_paths.append(wav_path)
                transcriptions.append(text)
            else:
                failure_counts[status] += 1
                if len(error_samples) < 5:
                    error_samples.append((status, result, futures[future][0]))
    
    print(f"\n✓ Successfully loaded {len(audio_paths)} audio files")
    print(f"\nFailure breakdown:")
    for reason, count in failure_counts.items():
        if count > 0:
            print(f"  {reason}: {count}")
    
    if error_samples:
        print(f"\nFirst few errors:")
        for status, error, path in error_samples:
            print(f"  {status}: {path}")
            if error:
                print(f"    Details: {error}")
    
    return audio_paths, transcriptions


def test_baseline_whisper(audio_paths, transcriptions, num_samples=5):
    print("\n" + "="*80)
    print("BASELINE WHISPER PERFORMANCE")
    print("="*80)
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for i in range(min(num_samples, len(audio_paths))):
        try:
            waveform, sr = torchaudio.load(audio_paths[i])
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            
            generated_ids = model.generate(input_features, min_length=10, length_penalty=2.0)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"\nSample {i+1}:")
            print(f"  Reference:  {transcriptions[i]}")
            print(f"  Prediction: {transcription}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("="*80 + "\n")


def main():
    # Force single GPU usage
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
    # ============================================================================
    # CHECKPOINT CONFIGURATION
    # ============================================================================
    # Set to None to start fresh, or provide path to resume from checkpoint
    RESUME_FROM_CHECKPOINT = ''
    
    csv_path = "common Voice Dataset/Transcriptions.csv"
    audio_root = "common Voice Dataset/Feebles"

    print("Loading dataset...")
    train_audio, train_texts = load_dataset_parallel(
        csv_path, 
        audio_root, 
        max_samples=5000,
        num_workers=20
    )
    
    train_audio, val_audio, train_texts, val_texts = train_test_split(
        train_audio, train_texts, test_size=0.05, random_state=42
    )
    
    print(f"Training samples: {len(train_audio)}")
    print(f"Validation samples: {len(val_audio)}")

    # Only test baseline if starting fresh
    if RESUME_FROM_CHECKPOINT is None:
        test_baseline_whisper(train_audio, train_texts, num_samples=5)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(1)}")

    config = ModelConfig()
    config.batch_size = 8
    config.num_epochs = 10
    config.learning_rate = 5e-6
    config.gradient_accumulation_steps = 2
    config.min_generation_length = 10
    config.length_penalty = 2.5
    config.warmup_steps = 100

    print("\nInitializing model...")
    trainer = HybridModelTrainer(config)
    trainer.model = trainer.model.to(device)
    
    # ============================================================================
    # LOAD FROM CHECKPOINT IF SPECIFIED
    # ============================================================================
    start_epoch = 0
    if RESUME_FROM_CHECKPOINT is not None:
        if os.path.exists(RESUME_FROM_CHECKPOINT):
            print("\n" + "="*80)
            print(f"LOADING CHECKPOINT: {RESUME_FROM_CHECKPOINT}")
            print("="*80)
            
            # Need to initialize optimizer and scheduler before loading
            trainer.model.unfreeze_decoder()
            
            fusion_params = []
            decoder_params = []
            
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    if 'fusion' in name:
                        fusion_params.append(param)
                    else:
                        decoder_params.append(param)
            
            trainer.optimizer = torch.optim.AdamW([
                {'params': fusion_params, 'lr': config.learning_rate},
                {'params': decoder_params, 'lr': config.learning_rate}
            ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
            
            # Create scheduler
            train_loader_temp = trainer.prepare_dataset(train_audio, train_texts)
            total_steps = len(train_loader_temp) * config.num_epochs // config.gradient_accumulation_steps
            
            from torch.optim.lr_scheduler import OneCycleLR
            trainer.scheduler = OneCycleLR(
                trainer.optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            # Load checkpoint
            checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state loaded")
            
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state loaded")
            
            start_epoch = checkpoint['epoch'] + 1
            
            if 'global_step' in checkpoint:
                trainer.global_step = checkpoint['global_step']
            
            print(f"✓ Checkpoint loaded successfully")
            print(f"✓ Resuming from epoch {start_epoch}")
            print(f"✓ Global step: {trainer.global_step}")
            print("="*80 + "\n")
        else:
            print(f"\n⚠ WARNING: Checkpoint not found at {RESUME_FROM_CHECKPOINT}")
            print("Starting training from scratch...\n")
    
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "="*80)
    print("STARTING PHASED TRAINING")
    if RESUME_FROM_CHECKPOINT:
        print(f"(Resuming from epoch {start_epoch})")
    print("="*80)
    
    # Modify fine_tune call to pass start_epoch
    trainer.fine_tune(
        train_audio_paths=train_audio,
        train_transcriptions=train_texts,
        val_audio_paths=val_audio,
        val_transcriptions=val_texts,
        start_epoch=start_epoch  # Pass the starting epoch
    )

    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    final_metrics = trainer.evaluate(val_audio, val_texts)
    print(f"\nFinal Results:")
    print(f"  WER: {final_metrics['wer']:.4f} ({final_metrics['wer']*100:.2f}%)")
    print(f"  CER: {final_metrics['cer']:.4f} ({final_metrics['cer']*100:.2f}%)")
    
    print("\n" + "="*80)
    print("FINAL SAMPLE PREDICTIONS")
    print("="*80)
    
    for i in range(min(10, len(final_metrics['predictions']))):
        ref = final_metrics['references'][i]
        pred = final_metrics['predictions'][i]
        
        ref_norm = ref.lower().strip().replace("'", "'")
        pred_norm = pred.lower().strip().replace("'", "'")
        match = "✓" if ref_norm == pred_norm else "✗"
        
        print(f"\n{i+1}. {match}")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
    
    results_df = pd.DataFrame({
        'reference': final_metrics['references'],
        'prediction': final_metrics['predictions']
    })
    results_path = Path(config.output_dir) / "final_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
