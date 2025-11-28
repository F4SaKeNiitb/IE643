/**
 * Real-Time Audio Transcription Frontend
 * Captures audio from microphone, splits into 5-second chunks,
 * sends via WebSocket, and displays transcription results
 */

class AudioTranscriptionApp {
    constructor() {
        // Configuration
        this.SAMPLE_RATE = 16000;
        this.CHUNK_DURATION = 5; // seconds
        this.CHUNK_SIZE = this.SAMPLE_RATE * this.CHUNK_DURATION;
        
        // State
        this.isRecording = false;
        this.isConnected = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.socket = null;
        this.clientId = this.generateClientId();
        
        // Audio buffer
        this.audioBuffer = [];
        this.chunkCount = 0;
        
        // DOM elements
        this.elements = {
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            resetBtn: document.getElementById('reset-btn'),
            hybridOutput: document.getElementById('hybrid-output'),
            baselineOutput: document.getElementById('baseline-output'),
            connectionIndicator: document.getElementById('connection-indicator'),
            connectionStatus: document.getElementById('connection-status'),
            recordingIndicator: document.getElementById('recording-indicator'),
            recordingStatus: document.getElementById('recording-status'),
            audioVisualizer: document.getElementById('audio-visualizer'),
            bufferSize: document.getElementById('buffer-size'),
            chunkCount: document.getElementById('chunk-count'),
            chunkDuration: document.getElementById('chunk-duration'),
            copyHybrid: document.getElementById('copy-hybrid'),
            copyBaseline: document.getElementById('copy-baseline'),
        };
        
        // Bind event handlers
        this.bindEvents();
        
        // Initialize WebSocket connection
        this.connectWebSocket();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.startRecording());
        this.elements.stopBtn.addEventListener('click', () => this.stopRecording());
        this.elements.resetBtn.addEventListener('click', () => this.resetSession());
        this.elements.copyHybrid.addEventListener('click', () => this.copyToClipboard('hybrid'));
        this.elements.copyBaseline.addEventListener('click', () => this.copyToClipboard('baseline'));
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = window.location.port || '8000';
        const wsUrl = `${protocol}//${host}:${port}/ws/${this.clientId}`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            
            // Send configuration
            this.socket.send(JSON.stringify({
                command: 'config',
                sample_rate: this.SAMPLE_RATE
            }));
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'transcription':
                this.updateTranscription(data);
                break;
            case 'reset':
                console.log('Session reset:', data.message);
                break;
            case 'config_ack':
                console.log('Configuration acknowledged:', data);
                this.elements.chunkDuration.textContent = `${data.chunk_duration}s`;
                break;
            default:
                console.log('Unknown message type:', data);
        }
    }
    
    updateTranscription(data) {
        // Update hybrid output
        if (data.hybrid) {
            this.elements.hybridOutput.innerHTML = this.formatTranscription(data.hybrid, data.chunk_hybrid);
            this.elements.hybridOutput.scrollTop = this.elements.hybridOutput.scrollHeight;
        }
        
        // Update baseline output
        if (data.baseline) {
            this.elements.baselineOutput.innerHTML = this.formatTranscription(data.baseline, data.chunk_baseline);
            this.elements.baselineOutput.scrollTop = this.elements.baselineOutput.scrollHeight;
        }
    }
    
    formatTranscription(fullText, newChunk) {
        if (!fullText) return '<span class="text-gray-500 italic">No transcription yet...</span>';
        
        // Highlight the new chunk if available
        if (newChunk && fullText.endsWith(newChunk)) {
            const previousText = fullText.slice(0, -newChunk.length).trim();
            return `${previousText} <span class="new-text-highlight">${newChunk}</span>`;
        }
        
        return fullText;
    }
    
    updateConnectionStatus(connected) {
        if (connected) {
            this.elements.connectionIndicator.className = 'status-indicator status-connected';
            this.elements.connectionStatus.textContent = 'Connected';
        } else {
            this.elements.connectionIndicator.className = 'status-indicator status-disconnected';
            this.elements.connectionStatus.textContent = 'Disconnected';
        }
    }
    
    updateRecordingStatus(recording) {
        if (recording) {
            this.elements.recordingIndicator.className = 'status-indicator status-recording recording-pulse';
            this.elements.recordingStatus.textContent = 'Recording';
            this.elements.audioVisualizer.classList.remove('hidden');
        } else {
            this.elements.recordingIndicator.className = 'status-indicator status-disconnected';
            this.elements.recordingStatus.textContent = 'Not Recording';
            this.elements.audioVisualizer.classList.add('hidden');
        }
    }
    
    async startRecording() {
        if (this.isRecording) return;
        if (!this.isConnected) {
            alert('Not connected to server. Please wait for connection.');
            return;
        }
        
        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.SAMPLE_RATE,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });
            
            // Create AudioContext
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.SAMPLE_RATE
            });
            
            // Create source from microphone stream
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create ScriptProcessor for audio processing
            // Note: ScriptProcessorNode is deprecated but more widely supported
            // We'll use it with a reasonable buffer size
            const bufferSize = 4096;
            const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            processor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                
                const inputData = event.inputBuffer.getChannelData(0);
                const audioData = new Float32Array(inputData);
                
                // Add to buffer
                this.audioBuffer.push(...audioData);
                this.elements.bufferSize.textContent = this.audioBuffer.length;
                
                // Check if we have enough samples for a chunk
                while (this.audioBuffer.length >= this.CHUNK_SIZE) {
                    const chunk = new Float32Array(this.audioBuffer.splice(0, this.CHUNK_SIZE));
                    this.sendAudioChunk(chunk);
                    this.chunkCount++;
                    this.elements.chunkCount.textContent = this.chunkCount;
                }
            };
            
            // Connect nodes
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            this.workletNode = processor;
            this.isRecording = true;
            
            // Update UI
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.updateRecordingStatus(true);
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Failed to access microphone. Please ensure you have granted permission.');
        }
    }
    
    sendAudioChunk(audioData) {
        if (!this.isConnected || !this.socket) return;
        
        // Convert Float32Array to bytes and send
        const buffer = audioData.buffer;
        this.socket.send(buffer);
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        // Disconnect and close audio context
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        // Send remaining buffer if substantial
        if (this.audioBuffer.length > this.SAMPLE_RATE) { // At least 1 second
            const remainingAudio = new Float32Array(this.audioBuffer);
            this.sendAudioChunk(remainingAudio);
        }
        
        // Clear buffer
        this.audioBuffer = [];
        this.elements.bufferSize.textContent = '0';
        
        // Update UI
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.updateRecordingStatus(false);
        
        console.log('Recording stopped');
    }
    
    resetSession() {
        // Stop recording if active
        if (this.isRecording) {
            this.stopRecording();
        }
        
        // Send reset command to server
        if (this.isConnected && this.socket) {
            this.socket.send(JSON.stringify({ command: 'reset' }));
        }
        
        // Reset local state
        this.audioBuffer = [];
        this.chunkCount = 0;
        
        // Reset UI
        this.elements.hybridOutput.innerHTML = '<span class="text-gray-500 italic">Your transcription will appear here...</span>';
        this.elements.baselineOutput.innerHTML = '<span class="text-gray-500 italic">Baseline transcription will appear here...</span>';
        this.elements.bufferSize.textContent = '0';
        this.elements.chunkCount.textContent = '0';
        
        console.log('Session reset');
    }
    
    copyToClipboard(type) {
        const element = type === 'hybrid' ? this.elements.hybridOutput : this.elements.baselineOutput;
        const text = element.innerText;
        
        if (text && !text.includes('will appear here')) {
            navigator.clipboard.writeText(text).then(() => {
                // Show feedback
                const button = type === 'hybrid' ? this.elements.copyHybrid : this.elements.copyBaseline;
                const originalHTML = button.innerHTML;
                button.innerHTML = 'Copied!';
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.transcriptionApp = new AudioTranscriptionApp();
});
