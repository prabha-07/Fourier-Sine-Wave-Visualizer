from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
from scipy.fft import fft, fftfreq
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Allow frontend to call API

# Configuration
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'webm'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB (reduced for faster processing)
MAX_COMPONENTS = 48
FFT_SAMPLE_LIMIT = 262144

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """
    Analyze uploaded audio file using librosa and scipy.
    Returns FFT data in JSON format.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use MP3, WAV, OGG, etc.'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB'}), 400
    
    # Save temporarily
    temp_path = None
    try:
        # Create temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Load audio using librosa with maximum optimization for speed
        print(f"Loading audio: {temp_path}")
        
        # Use very low sample rate for fastest processing (8000 Hz covers up to 4kHz)
        # This is sufficient for FFT visualization and dramatically faster
        TARGET_SR = 8000
        
        # Limit to first 15 seconds for very long files - critical for timeout prevention
        # Use fastest resampling method
        print("Starting librosa.load()...")
        y, sr = librosa.load(
            temp_path, 
            sr=TARGET_SR,  # Very low sample rate = much faster processing
            duration=15.0,  # Limit to first 15 seconds (critical for timeout)
            mono=True,  # Ensure mono
            res_type='kaiser_fast'  # Fastest resampling method
        )
        
        N = len(y)
        print(f"Audio loaded: {N} samples at {sr} Hz ({N/sr:.2f} seconds)")
        
        # Aggressively limit samples for FFT to ensure fast computation
        FFT_LIMIT = 65536  # Even smaller limit (2^16) for fastest FFT
        if N > FFT_LIMIT:
            print(f"Downsampling from {N} to {FFT_LIMIT} samples")
            step = N // FFT_LIMIT
            y = y[::step]
            N = len(y)
            print(f"After downsampling: {N} samples")
        
        print("Starting FFT computation...")
        
        # Compute Fourier Transform (your existing code!)
        Y = fft(y)
        freqs = fftfreq(N, 1/sr)
        
        # Only take positive frequencies (exclude DC)
        mask = freqs > 0
        freqs_positive = freqs[mask]
        Y_positive = Y[mask]
        magnitudes = np.abs(Y_positive)
        
        # Find top frequencies
        top_indices = np.argsort(magnitudes)[-MAX_COMPONENTS:][::-1]
        top_freqs = freqs_positive[top_indices]
        top_mags = magnitudes[top_indices]
        
        # Normalize amplitudes
        max_magnitude = np.max(top_mags) if len(top_mags) > 0 else 1
        
        # Create components list
        components = []
        for freq, mag in zip(top_freqs, top_mags):
            components.append({
                'freq': float(freq),
                'magnitude': float(mag),
                'amplitude': float(mag / max_magnitude)
            })
        
        # Sort by magnitude for top frequencies
        components_sorted = sorted(components, key=lambda x: x['magnitude'], reverse=True)
        top_frequencies = components_sorted[:12]  # Top 12 for table
        
        # Sort by frequency for visualization (bass to treble)
        visual_components = sorted(components_sorted[:MAX_COMPONENTS], key=lambda x: x['freq'])
        
        # Calculate FFT size (next power of 2)
        fft_size = 1
        while fft_size < N:
            fft_size <<= 1
        
        # Return data
        return jsonify({
            'success': True,
            'fileName': secure_filename(file.filename),
            'sampleRate': int(sr),
            'duration': float(N / sr),
            'fftSize': fft_size,
            'components': visual_components,
            'topFrequencies': top_frequencies,
            'stats': {
                'totalSamples': int(N),
                'nyquistFrequency': float(sr / 2),
                'frequencyResolution': float(sr / fft_size),
                'maxAmplitude': float(np.max(np.abs(y))),
                'meanAmplitude': float(np.mean(np.abs(y)))
            }
        })
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Fourier Visualizer API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print("=" * 60)
    print("Fourier Sine Wave Visualizer - Flask Backend")
    print("=" * 60)
    print(f"Starting server on http://localhost:{port}")
    print("=" * 60)
    app.run(debug=debug, host='0.0.0.0', port=port)