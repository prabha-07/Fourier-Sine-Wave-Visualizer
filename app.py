from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import librosa
import soundfile as sf
import numpy as np
from scipy.fft import fft, fftfreq
import tempfile
import os
import signal
import time
import subprocess
import atexit
import threading
import glob
import logging
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static', static_url_path='')

# Configure CORS - restrict to your domain in production
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, you might want to restrict this to your domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Use in-memory storage (for production, consider Redis)
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'webm'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
MAX_COMPONENTS = 48
FFT_SAMPLE_LIMIT = 262144
TEMP_DIR = tempfile.gettempdir()
CLEANUP_INTERVAL = 3600  # Clean up temp files every hour
TEMP_FILE_MAX_AGE = 1800  # Delete temp files older than 30 minutes

# Track active processing requests to prevent overload
active_requests = threading.Semaphore(3)  # Max 3 concurrent processing requests

def allowed_file(filename):
    """Validate file extension"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def cleanup_old_temp_files():
    """Clean up old temporary files to prevent disk space issues"""
    try:
        current_time = time.time()
        pattern = os.path.join(TEMP_DIR, 'tmp*')
        cleaned = 0
        
        for filepath in glob.glob(pattern):
            try:
                # Check if file is old enough to delete
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > TEMP_FILE_MAX_AGE:
                        os.unlink(filepath)
                        cleaned += 1
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not delete temp file {filepath}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old temporary files")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        cleanup_old_temp_files()

# Start background cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# Register cleanup on exit
atexit.register(cleanup_old_temp_files)

def decode_mp3_ffmpeg(mp3_path, wav_path, target_sr=22050, max_duration=None):
    """
    Decode MP3 to WAV using ffmpeg directly (much more memory-efficient than librosa).
    This avoids heavy in-RAM MP3 decode and is faster.
    If max_duration is None, processes the entire file.
    """
    logger.info(f"Decoding MP3 to WAV using ffmpeg: {mp3_path} -> {wav_path}")
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
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30  # 30 second timeout for ffmpeg
        )
        
        decode_time = time.time() - start_time
        logger.info(f"ffmpeg decode completed in {decode_time:.2f} seconds")
        
        # Verify the output file was created and get info
        if not os.path.exists(wav_path):
            raise Exception("ffmpeg completed but output WAV file was not created")
        
        # Get file info using soundfile (fast)
        info = sf.info(wav_path)
        num_samples = info.frames
        sr = info.samplerate
        
        logger.info(f"Decoded audio: {num_samples} samples at {sr} Hz ({num_samples/sr:.2f} seconds)")
        
        return True, num_samples, sr
        
    except subprocess.TimeoutExpired:
        logger.error("MP3 decoding timed out")
        raise Exception("MP3 decoding timed out. The file may be too large or complex. Please try a shorter file or convert to WAV format first.")
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.error(f"ffmpeg error: {error_output}")
        raise Exception(f"Failed to decode MP3 with ffmpeg: {error_output}. Please check the file format or convert to WAV first.")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error decoding MP3 with ffmpeg: {error_msg}", exc_info=True)
        raise Exception(f"Failed to process MP3: {error_msg}. Please try converting to WAV format first.")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")  # Limit to 10 requests per minute per IP
def analyze_audio():
    """
    Analyze uploaded audio file using librosa and scipy.
    Returns FFT data in JSON format.
    """
    # Check if we can handle another concurrent request
    if not active_requests.acquire(blocking=False):
        logger.warning(f"Request rejected: too many concurrent processing requests from {get_remote_address()}")
        return jsonify({'error': 'Server is busy processing other requests. Please try again in a moment.'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate filename security
        filename = secure_filename(file.filename)
        if not filename or filename != file.filename:
            logger.warning(f"Invalid filename detected: {file.filename}")
            return jsonify({'error': 'Invalid filename. Please use a valid audio file name.'}), 400
        
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type. Use MP3, WAV, OGG, etc.'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB'}), 400
        
        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400
        
        logger.info(f"Processing audio file: {filename} ({file_size / 1024 / 1024:.2f}MB) from {get_remote_address()}")
    
        # Save temporarily
        temp_path = None
        original_temp_path = None
        wav_path = None
        try:
            # Create temp file with secure naming
            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
            
            # Use higher sample rate for better frequency resolution (22050 Hz covers up to 11kHz)
            # This allows us to see frequencies up to 4000+ Hz range properly
            TARGET_SR = 22050
            MAX_DURATION = None  # Process entire audio file for full frequency analysis
            
            # Convert MP3 to WAV first, then process WAV (much faster)
            logger.info(f"Processing audio: {temp_path}")
            file_ext = os.path.splitext(filename)[1].lower()
            
            original_temp_path = temp_path
            
            # If MP3, decode to WAV using ffmpeg (much more efficient than librosa)
            if file_ext == '.mp3':
                logger.info("MP3 file detected - decoding to WAV using ffmpeg...")
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
                logger.info(f"Using decoded WAV file: {wav_path}")
            
            # Now load audio using fast soundfile method (works for WAV, FLAC, and converted MP3)
            logger.info("Loading audio with soundfile (fast method)...")
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
                logger.warning(f"soundfile failed: {e}, falling back to librosa")
                # Fallback to librosa for other formats (loads entire file)
                y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True, res_type='kaiser_fast')
            
            load_time = time.time() - start_time
            logger.info(f"Audio load completed in {load_time:.2f} seconds")
            
            N = len(y)
            logger.info(f"Audio loaded: {N} samples at {sr} Hz ({N/sr:.2f} seconds)")
            
            # Limit samples for FFT if needed (original limit for good frequency resolution)
            FFT_LIMIT = 262144  # Original limit (2^18) for good frequency resolution
            if N > FFT_LIMIT:
                logger.info(f"Downsampling from {N} to {FFT_LIMIT} samples")
                step = N // FFT_LIMIT
                y = y[::step]
                N = len(y)
                logger.info(f"After downsampling: {N} samples")
            
            logger.info("Starting FFT computation...")
            fft_start = time.time()
            
            # Compute Fourier Transform (your existing code!)
            Y = fft(y)
            fft_time = time.time() - fft_start
            logger.info(f"FFT computation completed in {fft_time:.2f} seconds")
            
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
            
            logger.info(f"Successfully processed audio file: {filename}")
            
            # Return data
            return jsonify({
                'success': True,
                'fileName': secure_filename(filename),
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
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            
            # Provide helpful error message for timeout issues
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'too long' in error_msg.lower():
                error_msg = f"Processing took too long. MP3 files can be slow to process. Try converting to WAV format for faster processing, or use a shorter audio file (under 5 seconds)."
            
            return jsonify({'error': f'Error processing audio: {error_msg}'}), 500
        
        finally:
            # Clean up temp files (both original and converted WAV if MP3)
            cleanup_files = [temp_path, original_temp_path, wav_path]
            for file_path in cleanup_files:
                if file_path and os.path.exists(file_path) and file_path != temp_path:
                    try:
                        os.unlink(file_path)
                        logger.debug(f"Cleaned up temp file: {file_path}")
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not delete temp file {file_path}: {e}")
            
            # Always clean up the main temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up main temp file: {temp_path}")
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not delete main temp file {temp_path}: {e}")
    
    except Exception as e:
        # Handle any errors that occur before the inner try block
        logger.error(f"Error in analyze_audio: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    
    finally:
        # Always release the semaphore (even if error occurs before inner try)
        active_requests.release()

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with system status"""
    try:
        # Check available disk space in temp directory
        import shutil
        disk_usage = shutil.disk_usage(TEMP_DIR)
        free_space_gb = disk_usage.free / (1024**3)
        
        return jsonify({
            'status': 'ok',
            'message': 'Fourier Visualizer API is running',
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'active_requests': active_requests._value if hasattr(active_requests, '_value') else 'unknown',
                'available_disk_space_gb': round(free_space_gb, 2),
                'temp_directory': TEMP_DIR
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'ok',
            'message': 'Fourier Visualizer API is running',
            'timestamp': datetime.utcnow().isoformat()
        }), 200

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Too many requests. Please wait a moment before trying again.',
        'retry_after': getattr(e, 'retry_after', 60)
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({
        'error': 'An internal server error occurred. Please try again later.'
    }), 500

@app.errorhandler(413)
def request_too_large_handler(e):
    """Handle request too large errors"""
    return jsonify({
        'error': f'File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB.'
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print("=" * 60)
    print("Fourier Sine Wave Visualizer - Flask Backend")
    print("=" * 60)
    print(f"Starting server on http://localhost:{port}")
    print("=" * 60)
    app.run(debug=debug, host='0.0.0.0', port=port)