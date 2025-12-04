from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import soundfile as sf
import numpy as np
from scipy.fft import fft, fftfreq
import tempfile
import os
import signal
import time
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Allow frontend to call API

# Configuration
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'webm'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (restored original limit)
MAX_COMPONENTS = 48
FFT_SAMPLE_LIMIT = 262144

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_mp3_ffmpeg(mp3_path, wav_path, target_sr=22050, max_duration=None):
    """
    Decode MP3 to WAV using ffmpeg directly (much more memory-efficient than librosa).
    This avoids heavy in-RAM MP3 decode and is faster.
    If max_duration is None, processes the entire file.
    """
    print(f"Decoding MP3 to WAV using ffmpeg: {mp3_path} -> {wav_path}")
    start_time = time.time()
    
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise Exception("ffmpeg is not installed or not available. MP3 support requires ffmpeg. Please convert your MP3 to WAV format first.")
        
        # Use ffmpeg to decode MP3 directly to WAV with desired settings
        cmd = [
            "ffmpeg", "-y",  # -y to overwrite output file
            "-i", mp3_path,  # input file
            "-ac", "1",      # mono (1 audio channel)
            "-ar", str(target_sr),  # sample rate
            "-f", "wav",     # output format
            wav_path         # output file
        ]
        
        # Only add duration limit if specified
        if max_duration is not None:
            cmd.insert(-2, "-t")  # Insert before output format
            cmd.insert(-2, str(max_duration))  # Insert duration value
        
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30  # 30 second timeout for ffmpeg
        )
        
        decode_time = time.time() - start_time
        print(f"ffmpeg decode completed in {decode_time:.2f} seconds")
        
        # Verify the output file was created and get info
        if not os.path.exists(wav_path):
            raise Exception("ffmpeg completed but output WAV file was not created")
        
        # Get file info using soundfile (fast)
        info = sf.info(wav_path)
        num_samples = info.frames
        sr = info.samplerate
        
        print(f"Decoded audio: {num_samples} samples at {sr} Hz ({num_samples/sr:.2f} seconds)")
        
        return True, num_samples, sr
        
    except subprocess.TimeoutExpired:
        raise Exception("MP3 decoding timed out. The file may be too large or complex. Please try a shorter file or convert to WAV format first.")
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        print(f"ffmpeg error: {error_output}")
        raise Exception(f"Failed to decode MP3 with ffmpeg: {error_output}. Please check the file format or convert to WAV first.")
    except Exception as e:
        error_msg = str(e)
        print(f"Error decoding MP3 with ffmpeg: {error_msg}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to process MP3: {error_msg}. Please try converting to WAV format first.")

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
    
    # Note: With ffmpeg, we can handle larger files efficiently
    
    # Save temporarily
    temp_path = None
    original_temp_path = None
    wav_path = None
    try:
        # Create temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Convert MP3 to WAV first, then process WAV (much faster)
        print(f"Processing audio: {temp_path}")
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Use higher sample rate for better frequency resolution (22050 Hz covers up to 11kHz)
        # This allows us to see frequencies up to 4000+ Hz range properly
        TARGET_SR = 22050
        MAX_DURATION = None  # Process entire audio file for full frequency analysis
        
        original_temp_path = temp_path
        
        # If MP3, decode to WAV using ffmpeg (much more efficient than librosa)
        if file_ext == '.mp3':
            print("MP3 file detected - decoding to WAV using ffmpeg...")
            wav_path = temp_path.replace('.mp3', '.wav').replace('.MP3', '.wav')
            
            # Decode MP3 to WAV using ffmpeg (memory-efficient, processes entire file)
            conversion_start = time.time()
            success, num_samples, sr = decode_mp3_ffmpeg(temp_path, wav_path, TARGET_SR, None)
            conversion_time = time.time() - conversion_start
            
            if not success:
                raise Exception("Failed to decode MP3 to WAV. ffmpeg may not be installed. Please convert your MP3 to WAV format first using an online converter or audio software (like Audacity, VLC, or ffmpeg). WAV files work immediately without additional dependencies!")
            
            if conversion_time > 25:
                raise Exception(f"MP3 decoding took too long ({conversion_time:.2f}s). Please try a shorter file or convert to WAV format first.")
            
            # Use the WAV file for processing
            temp_path = wav_path
            print(f"Using decoded WAV file: {wav_path}")
        
        # Now load audio using fast soundfile method (works for WAV, FLAC, and converted MP3)
        print("Loading audio with soundfile (fast method)...")
        start_time = time.time()
        
        try:
            # Use soundfile for fast loading (works for WAV, FLAC, and our converted WAV)
            data, sr = sf.read(temp_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample to target sample rate if needed
            if sr != TARGET_SR:
                import scipy.signal
                num_samples = int(len(data) * TARGET_SR / sr)
                data = scipy.signal.resample(data, num_samples)
                sr = TARGET_SR
            
            y = data
            
            # Process entire file - no duration limit
            
        except Exception as e:
            print(f"soundfile failed: {e}, falling back to librosa")
            # Fallback to librosa for other formats (loads entire file)
            y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True, res_type='kaiser_fast')
        
        load_time = time.time() - start_time
        print(f"Audio load completed in {load_time:.2f} seconds")
        
        N = len(y)
        print(f"Audio loaded: {N} samples at {sr} Hz ({N/sr:.2f} seconds)")
        
        # Limit samples for FFT if needed (original limit for good frequency resolution)
        FFT_LIMIT = 262144  # Original limit (2^18) for good frequency resolution
        if N > FFT_LIMIT:
            print(f"Downsampling from {N} to {FFT_LIMIT} samples")
            step = N // FFT_LIMIT
            y = y[::step]
            N = len(y)
            print(f"After downsampling: {N} samples")
        
        print("Starting FFT computation...")
        fft_start = time.time()
        
        # Compute Fourier Transform (your existing code!)
        Y = fft(y)
        fft_time = time.time() - fft_start
        print(f"FFT computation completed in {fft_time:.2f} seconds")
        
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
        
        # Provide helpful error message for timeout issues
        error_msg = str(e)
        if 'timeout' in error_msg.lower() or 'too long' in error_msg.lower():
            error_msg = f"Processing took too long. MP3 files can be slow to process. Try converting to WAV format for faster processing, or use a shorter audio file (under 5 seconds)."
        
        return jsonify({'error': f'Error processing audio: {error_msg}'}), 500
    
    finally:
        # Clean up temp files (both original and converted WAV if MP3)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        # Clean up original MP3 file if we converted it
        if original_temp_path and original_temp_path != temp_path and os.path.exists(original_temp_path):
            try:
                os.unlink(original_temp_path)
            except:
                pass
        # Clean up converted WAV file if it exists separately
        if wav_path and wav_path != temp_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
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