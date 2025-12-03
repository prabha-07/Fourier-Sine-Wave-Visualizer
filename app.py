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

def convert_mp3_to_wav(mp3_path, wav_path, target_sr=8000, max_duration=2.0):
    """
    Convert MP3 file to WAV format for faster processing.
    Uses most aggressive settings to minimize conversion time.
    Note: MP3 decoding is inherently slow - consider converting to WAV client-side for best performance.
    """
    print(f"Converting MP3 to WAV: {mp3_path} -> {wav_path}")
    start_time = time.time()
    
    try:
        # Load MP3 directly at target sample rate (faster than loading native then resampling)
        # Use most aggressive settings: very low SR, very short duration, fastest resampling
        print(f"Loading MP3 at {target_sr}Hz, max {max_duration}s (aggressive settings for speed)...")
        y, sr = librosa.load(
            mp3_path,
            sr=target_sr,  # Load directly at target SR (faster than resampling after)
            duration=max_duration,  # Very short duration (2 seconds) to stay under timeout
            mono=True,
            res_type='kaiser_fast'  # Fastest resampling
        )
        
        load_time = time.time() - start_time
        print(f"MP3 loaded in {load_time:.2f} seconds, {len(y)} samples")
        
        # Safety check - if loading took too long, abort early
        if load_time > 15:
            raise Exception(f"MP3 loading took too long ({load_time:.2f}s). MP3 files are slow to process on this server. Please convert to WAV format first for faster processing.")
        
        # Save as WAV using soundfile (very fast)
        print("Saving as WAV...")
        save_start = time.time()
        sf.write(wav_path, y, sr, format='WAV', subtype='PCM_16')
        save_time = time.time() - save_start
        
        convert_time = time.time() - start_time
        print(f"MP3 to WAV conversion completed in {convert_time:.2f} seconds (load: {load_time:.2f}s, save: {save_time:.2f}s)")
        print(f"Converted audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f} seconds)")
        
        return True, len(y), sr
        
    except Exception as e:
        print(f"Error converting MP3 to WAV: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None

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
    
    # Reject very large files early (MP3 decoding is slow for large files)
    if file_size > 5 * 1024 * 1024:  # 5MB
        return jsonify({'error': 'File too large for processing. Please use a file under 5MB for faster processing.'}), 400
    
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
        
        # Use very low sample rate for fastest processing (8000 Hz covers up to 4kHz)
        TARGET_SR = 8000
        MAX_DURATION = 2.0  # Limit to 2 seconds for MP3 (very aggressive for timeout prevention)
        
        original_temp_path = temp_path
        
        # If MP3, convert to WAV first (with timeout protection)
        if file_ext == '.mp3':
            print("MP3 file detected - converting to WAV for faster processing...")
            wav_path = temp_path.replace('.mp3', '.wav').replace('.MP3', '.wav')
            
            # Add timeout protection for MP3 conversion
            conversion_start = time.time()
            success, num_samples, sr = convert_mp3_to_wav(temp_path, wav_path, TARGET_SR, MAX_DURATION)
            conversion_time = time.time() - conversion_start
            
            if not success:
                raise Exception("Failed to convert MP3 to WAV. MP3 files are slow to process on this server. Please convert your MP3 to WAV format first using an online converter or audio software (like Audacity, VLC, or ffmpeg). WAV files process much faster!")
            
            if conversion_time > 20:
                raise Exception(f"MP3 conversion took too long ({conversion_time:.2f}s). MP3 decoding is slow on this server. For best results, please convert your MP3 to WAV format first. You can use online converters or tools like Audacity/VLC.")
            
            # Use the WAV file for processing
            temp_path = wav_path
            print(f"Using converted WAV file: {wav_path}")
        
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
            
            # Limit duration (safety check)
            max_samples = int(TARGET_SR * MAX_DURATION)
            if len(y) > max_samples:
                y = y[:max_samples]
                print(f"Truncated to {MAX_DURATION} seconds")
            
        except Exception as e:
            print(f"soundfile failed: {e}, falling back to librosa")
            # Fallback to librosa for other formats
            y, sr = librosa.load(temp_path, sr=TARGET_SR, duration=MAX_DURATION, mono=True, res_type='kaiser_fast')
        
        load_time = time.time() - start_time
        print(f"Audio load completed in {load_time:.2f} seconds")
        
        N = len(y)
        print(f"Audio loaded: {N} samples at {sr} Hz ({N/sr:.2f} seconds)")
        
        # Aggressively limit samples for FFT to ensure fast computation
        FFT_LIMIT = 32768  # Even smaller limit (2^15) for fastest FFT
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