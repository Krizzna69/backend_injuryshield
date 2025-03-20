# Gunicorn config for Render
import multiprocessing

# Use minimal workers on Render's free/starter plans
workers = 1

# Increase timeouts for video processing
timeout = 300  # 5 minutes
keepalive = 5

# Configure lower memory usage
worker_class = "gthread"
threads = 2
max_requests = 10
max_requests_jitter = 5

# Add logging
loglevel = "info"
accesslog = "-"
errorlog = "-"