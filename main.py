"""
FastAPI WebSocket backend for real-time audio transcription
"""
import sys
from pathlib import Path

# Add parent directory to path BEFORE any other imports
_parent_dir = str(Path(__file__).parent.parent.resolve())
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import asyncio
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from collections import deque
import struct
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Import from parent directory - must be after sys.path modification
import backpropnetwork
from backpropnetwork import HybridWav2VecWhisperModel, ModelConfig


class StreamingTranscriber:
    """Handles audio transcription with context window"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', chunk_duration: float = 5.0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * chunk_duration)
        
        print(f"Loading model on {self.device}...")
        
        # Load configuration
        self.config = ModelConfig()
        
        # Load processors
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            self.config.wav2vec_model
        )
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model
        )
        
        # Initialize hybrid model
        self.model = HybridWav2VecWhisperModel(self.config)
        
        # Load checkpoint - handle potential module path issues
        # Register ModelConfig in sys.modules under alternate paths that might be used
        import sys
        sys.modules['__main__'].ModelConfig = ModelConfig
        sys.modules['backpropnetwork'] = backpropnetwork
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load baseline Whisper for comparison
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.config.whisper_model
        ).to(self.device)
        self.whisper_model.eval()
        
        print(f"✓ Streaming model ready (chunk size: {chunk_duration}s)")
    
    def preprocess_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Preprocess single audio chunk"""
        try:
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            
            if len(audio_chunk) == 0:
                return np.zeros(self.chunk_size, dtype=np.float32)
            
            audio_chunk = audio_chunk.flatten()
            audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            max_val = np.abs(audio_chunk).max()
            if max_val > 1e-6:
                audio_chunk = audio_chunk / max_val
            else:
                audio_chunk = np.zeros_like(audio_chunk)
            
            # Pad or truncate to chunk_size
            if len(audio_chunk) < self.chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
            elif len(audio_chunk) > self.chunk_size:
                audio_chunk = audio_chunk[:self.chunk_size]
            
            return audio_chunk
            
        except Exception as e:
            print(f"Error in preprocess_chunk: {e}")
            return np.zeros(self.chunk_size, dtype=np.float32)
    
    def transcribe_chunk(self, audio_chunk: np.ndarray, context_buffer: deque, use_context: bool = True) -> tuple:
        """Transcribe single audio chunk"""
        try:
            waveform = self.preprocess_chunk(audio_chunk)
            
            # Skip if silence
            if np.abs(waveform).max() < 0.01:
                return "", ""
            
            # Add to context buffer
            context_buffer.append(waveform)
            
            # Use context window if available
            if use_context and len(context_buffer) > 1:
                context_window = np.concatenate(list(context_buffer))
            else:
                context_window = waveform
            
            # Process for Wav2Vec2 and Whisper
            wav2vec_inputs = self.wav2vec_processor(
                context_window,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            whisper_inputs = self.whisper_processor(
                context_window,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            wav2vec_input_values = wav2vec_inputs.input_values.to(self.device)
            whisper_input_features = whisper_inputs.input_features.to(self.device)
            
            # Hybrid model transcription
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    hybrid_outputs = self.model(
                        wav2vec_input_values=wav2vec_input_values,
                        whisper_input_features=whisper_input_features
                    )
            
            hybrid_generated_ids = hybrid_outputs['generated_ids']
            hybrid_text = self.whisper_processor.batch_decode(
                hybrid_generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # Baseline Whisper transcription
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    baseline_generated_ids = self.whisper_model.generate(
                        whisper_input_features,
                        max_length=448
                    )
            
            baseline_text = self.whisper_processor.batch_decode(
                baseline_generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return hybrid_text, baseline_text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", ""


# Global transcriber instance
transcriber = None


def get_transcriber():
    """Get or initialize the transcriber"""
    global transcriber
    if transcriber is None:
        # Look for checkpoint in parent directory
        base_path = Path(__file__).parent / ".."
        checkpoint_path = base_path / "checkpoints_tiny" / "checkpoint_latest_epoch_9.pt"
        checkpoint_path = checkpoint_path.resolve()
        
        if not checkpoint_path.exists():
            # Try alternative paths
            alt_path = base_path / "checkpoints" / "checkpoint_best_epoch_9.pt"
            if alt_path.resolve().exists():
                checkpoint_path = alt_path.resolve()
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading checkpoint from: {checkpoint_path}")
        transcriber = StreamingTranscriber(str(checkpoint_path), device)
    return transcriber


# FastAPI app
app = FastAPI(title="Real-Time Audio Transcription API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = {
            "websocket": websocket,
            "audio_buffer": np.array([], dtype=np.float32),
            "context_buffer": deque(maxlen=3),  # Keep last 3 chunks for context
            "accumulated_hybrid": "",
            "accumulated_baseline": "",
        }
        print(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected")
    
    async def send_transcription(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id]["websocket"].send_json(data)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    print("Initializing transcription model...")
    get_transcriber()
    print("Model ready!")


@app.get("/")
async def root():
    """Serve the frontend"""
    frontend_path = Path(__file__).parent / "index.html"
    frontend_path = frontend_path.resolve()
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Real-Time Audio Transcription API", "status": "running"}


@app.get("/app.js")
async def serve_js():
    """Serve the frontend JavaScript"""
    js_path = Path(__file__).parent / "app.js"
    js_path = js_path.resolve()
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    return {"error": "JavaScript file not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": transcriber is not None}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for streaming audio"""
    await manager.connect(websocket, client_id)
    
    trans = get_transcriber()
    chunk_size = trans.chunk_size
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive()
            
            if "bytes" in data:
                # Binary audio data (raw PCM float32 or int16)
                audio_bytes = data["bytes"]
                
                # Try to detect format: check if it's float32 or int16
                # Frontend sends float32 by default
                try:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                except:
                    # Fallback to int16
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to buffer
                conn = manager.active_connections[client_id]
                conn["audio_buffer"] = np.concatenate([conn["audio_buffer"], audio_data])
                
                # Process complete 5-second chunks
                while len(conn["audio_buffer"]) >= chunk_size:
                    chunk = conn["audio_buffer"][:chunk_size]
                    conn["audio_buffer"] = conn["audio_buffer"][chunk_size:]
                    
                    # Transcribe
                    hybrid_text, baseline_text = trans.transcribe_chunk(
                        chunk, 
                        conn["context_buffer"],
                        use_context=True
                    )
                    
                    if hybrid_text:
                        conn["accumulated_hybrid"] += " " + hybrid_text
                    if baseline_text:
                        conn["accumulated_baseline"] += " " + baseline_text
                    
                    # Send transcription result
                    await manager.send_transcription(client_id, {
                        "type": "transcription",
                        "hybrid": conn["accumulated_hybrid"].strip(),
                        "baseline": conn["accumulated_baseline"].strip(),
                        "chunk_hybrid": hybrid_text,
                        "chunk_baseline": baseline_text,
                    })
            
            elif "text" in data:
                # Handle text commands
                message = json.loads(data["text"])
                
                if message.get("command") == "reset":
                    # Reset buffers
                    conn = manager.active_connections[client_id]
                    conn["audio_buffer"] = np.array([], dtype=np.float32)
                    conn["context_buffer"].clear()
                    conn["accumulated_hybrid"] = ""
                    conn["accumulated_baseline"] = ""
                    
                    await manager.send_transcription(client_id, {
                        "type": "reset",
                        "message": "Session reset successfully"
                    })
                
                elif message.get("command") == "config":
                    # Handle configuration updates
                    sample_rate = message.get("sample_rate", 16000)
                    await manager.send_transcription(client_id, {
                        "type": "config_ack",
                        "sample_rate": sample_rate,
                        "chunk_duration": trans.chunk_duration
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)


# Mount frontend static files
frontend_dir = Path(__file__).parent / ".." / "frontend"
frontend_dir = frontend_dir.resolve()
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
