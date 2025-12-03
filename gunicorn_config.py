# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker to avoid memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 120  # 2 minutes timeout
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

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

