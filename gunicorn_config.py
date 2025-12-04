# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker to avoid memory issues with audio processing
worker_class = "gthread"  # Use threads for better concurrency
threads = 2  # 2 threads per worker
worker_connections = 1000
timeout = 300  # 5 minutes timeout (increased for large file processing)
keepalive = 5
graceful_timeout = 30  # Time to wait for workers to finish on restart

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "fourier_visualizer"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Worker recycling (helps prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Preload app for better performance
preload_app = False  # Set to False to avoid issues with multiprocessing

# Limit request line size to prevent abuse
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

