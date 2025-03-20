import os
import uuid
import re
import threading
import queue
import time
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from werkzeug.utils import secure_filename
import analysis
from flask_cors import CORS

# Create a custom task queue for processing videos
processing_queue = queue.Queue(maxsize=10)
processing_status = {}  # Track processing status

app = Flask(__name__, static_folder='static')
app.secret_key = "running_injury_analysis_secret_key"
CORS(app)  # Enable CORS for all routes

# Configure upload and results folders
BASE_DIR = r"backend/static"
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # Reduced to 200MB max upload size

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# Worker thread function to process videos asynchronously
def process_video_worker():
    """Worker thread to process videos from the queue"""
    print("Starting worker thread")
    while True:
        try:
            # Get job from queue with a timeout to allow checks
            try:
                job = processing_queue.get(timeout=1)
            except queue.Empty:
                # No job available, just continue the loop
                continue

            if job is None:  # None is a signal to stop the thread
                break

            video_path, analysis_dir, analysis_id, quality = job

            # Update status
            processing_status[analysis_id] = {"status": "processing", "progress": 0}

            try:
                # Process the video
                print(f"Processing video for analysis ID: {analysis_id}")
                result, risk_scores, frame_indices = analysis.analyze_running_form(
                    video_path, analysis_dir, quality=quality)

                # Save risk plot
                plot_path = analysis.save_risk_plot(risk_scores, analysis_dir)

                # Save report
                report_path = analysis.save_report(result, analysis_dir)

                # Update status
                processing_status[analysis_id] = {"status": "completed", "progress": 100}

                print(f"Processing completed for analysis ID: {analysis_id}")

                # Clean up original video to save space
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception as e:
                    print(f"Error removing video: {e}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                processing_status[analysis_id] = {"status": "error", "error": str(e)}
                print(f"Processing error for analysis ID {analysis_id}: {e}")

        except Exception as e:
            print(f"Error in worker thread: {e}")
        finally:
            # Mark task as done regardless of outcome
            if 'job' in locals() and job is not None:
                processing_queue.task_done()


# Start worker thread
worker_thread = threading.Thread(target=process_video_worker, daemon=True)
worker_thread.start()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'queue_size': processing_queue.qsize(),
        'active_jobs': len(processing_status)
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint for uploading videos and queuing for analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['video']
    quality = request.form.get('quality', 'low')  # Default to low quality for speed

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Create a unique directory for this analysis
            analysis_id = str(uuid.uuid4())
            analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
            os.makedirs(analysis_dir, exist_ok=True)

            # Save uploaded file with secure filename
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Queue the video for processing instead of processing immediately
            processing_queue.put((video_path, analysis_dir, analysis_id, quality))

            # Set initial status
            processing_status[analysis_id] = {"status": "queued", "progress": 0}

            print(f"Video queued for analysis, ID: {analysis_id}")

            # Return the analysis ID immediately
            return jsonify({'success': True, 'analysis_id': analysis_id})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({
            'error': 'File type not allowed. Please upload a video file (mp4, mov, avi, mkv, webm).'
        }), 400


@app.route('/api/status/<analysis_id>', methods=['GET'])
def check_status(analysis_id):
    """Check the status of video processing"""
    if analysis_id in processing_status:
        return jsonify(processing_status[analysis_id])
    else:
        # Check if the results already exist (processing completed)
        analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
        report_file = os.path.join(analysis_dir, 'analysis_report.txt')

        if os.path.exists(report_file):
            return jsonify({"status": "completed", "progress": 100})

        return jsonify({"status": "not_found"}), 404


@app.route('/api/results/<analysis_id>', methods=['GET'])
def get_analysis_results(analysis_id):
    """API endpoint for getting analysis results in JSON format"""
    try:
        analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)

        # Check if the analysis directory exists
        if not os.path.exists(analysis_dir):
            return jsonify({'error': 'Analysis not found'}), 404

        # Get the report file
        report_file = os.path.join(analysis_dir, 'analysis_report.txt')
        if not os.path.exists(report_file):
            # Check if it's still processing
            if analysis_id in processing_status:
                return jsonify({
                    'status': 'processing',
                    'progress': processing_status[analysis_id].get('progress', 0)
                }), 202
            return jsonify({'error': 'Analysis report not found'}), 404

        # Parse the report
        with open(report_file, 'r') as f:
            report_content = f.read()

        # Extract data from report
        risk_factors = []
        recommendations = []
        overall_risk = "Not Available"
        peak_risk = "Not Available"

        parsing_section = None
        for line in report_content.split('\n'):
            if 'Overall Risk Level:' in line:
                overall_risk = line.split('Overall Risk Level:')[1].strip()
            elif 'Peak Risk Level:' in line:
                peak_risk = line.split('Peak Risk Level:')[1].strip()
            elif 'Identified Risk Factors:' in line:
                parsing_section = 'risk_factors'
                continue
            elif 'Recommendations:' in line:
                parsing_section = 'recommendations'
                continue
            elif line.startswith('===') or line.strip() == '' or line.startswith('Analyzed video:'):
                parsing_section = None
                continue

            if parsing_section == 'risk_factors' and line.startswith('- '):
                risk_factors.append(line[2:])
            elif parsing_section == 'recommendations' and line.strip() and ':' not in line:
                if '. ' in line:
                    recommendations.append(line.split('. ', 1)[1])
                else:
                    recommendations.append(line)

        # Create URLs for the files - absolute URLs for React
        base_url = request.host_url.rstrip('/')
        video_path = f"{base_url}/api/video/{analysis_id}"
        plot_path = f"{base_url}/api/file/{analysis_id}/risk_over_time.png"

        # Return the results as JSON
        return jsonify({
            'analysis_id': analysis_id,
            'overall_risk': overall_risk,
            'peak_risk': peak_risk,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'video_path': video_path,
            'plot_path': plot_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/<analysis_id>', methods=['GET'])
def list_analysis_files(analysis_id):
    """List all files in the analysis directory (for debugging)"""
    analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)

    if not os.path.exists(analysis_dir):
        return jsonify({'error': 'Analysis directory not found'}), 404

    files = []
    try:
        for fname in os.listdir(analysis_dir):
            file_path = os.path.join(analysis_dir, fname)
            file_info = {
                'name': fname,
                'size': os.path.getsize(file_path),
                'modified': os.path.getmtime(file_path)
            }
            files.append(file_info)

        return jsonify({
            'analysis_id': analysis_id,
            'directory': analysis_dir,
            'files': files
        })
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500


@app.route('/api/video/<analysis_id>', methods=['GET'])
def stream_video(analysis_id):
    """Stream video with proper support for range requests"""
    analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)

    # Try multiple possible filenames for the video
    possible_filenames = ['analyzed_video.mp4', 'compressed_video.mp4', 'video.mp4']

    video_path = None
    for fname in possible_filenames:
        path = os.path.join(analysis_dir, fname)
        if os.path.exists(path):
            video_path = path
            break

    if not video_path:
        return jsonify({'error': 'Video not found'}), 404

    file_size = os.path.getsize(video_path)

    # Handle range header for video streaming
    range_header = request.headers.get('Range', None)
    if not range_header:
        # If no range is requested, send the entire file
        return send_file(video_path, mimetype='video/mp4')

    # Parse byte range from request header
    byte1, byte2 = 0, None
    match = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if match:
        groups = match.groups()
        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

    if byte2 is None:
        byte2 = file_size - 1

    length = byte2 - byte1 + 1

    # For large ranges, limit chunk size to avoid memory issues
    max_chunk = 1024 * 1024  # 1MB max chunk
    if length > max_chunk:
        byte2 = byte1 + max_chunk - 1
        length = max_chunk

    # Read the specified byte range
    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    # Set response headers for partial content
    headers = {
        'Content-Range': f'bytes {byte1}-{byte2}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': 'video/mp4',
        'Cache-Control': 'no-cache'
    }

    return Response(data, status=206, headers=headers)


@app.route('/api/file/<analysis_id>/<filename>')
def serve_analysis_file(analysis_id, filename):
    """API endpoint for serving static files from analysis directory"""
    analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)

    if not os.path.exists(os.path.join(analysis_dir, filename)):
        return jsonify({'error': f'File {filename} not found'}), 404

    return send_from_directory(analysis_dir, filename)


@app.route('/api/download/<analysis_id>/<file_type>')
def download_analysis_file(analysis_id, file_type):
    """API endpoint for downloading files"""
    analysis_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)

    if not os.path.exists(analysis_dir):
        return jsonify({'error': 'Analysis not found'}), 404

    if file_type == 'video':
        # Try multiple possible filenames for the video
        possible_filenames = ['analyzed_video.mp4', 'compressed_video.mp4', 'video.mp4']

        # Find the first matching file
        filename = None
        for fname in possible_filenames:
            if os.path.exists(os.path.join(analysis_dir, fname)):
                filename = fname
                break

        # If no matching file found, try to find any MP4 file
        if not filename:
            for fname in os.listdir(analysis_dir):
                if fname.endswith('.mp4'):
                    filename = fname
                    break

        if not filename:
            return jsonify({'error': 'Video file not found in analysis directory'}), 404

    elif file_type == 'report':
        filename = 'analysis_report.txt'
    elif file_type == 'plot':
        filename = 'risk_over_time.png'
    else:
        return jsonify({'error': 'Invalid file type'}), 400

    file_path = os.path.join(analysis_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': f'File {filename} not found'}), 404

    # Print debug information
    print(f"Sending file: {file_path}")

    try:
        return send_from_directory(analysis_dir, filename, as_attachment=True)
    except Exception as e:
        print(f"Error sending file: {e}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500


# Serve React app - All other routes will be handled by React Router
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    # Check if path is for an API endpoint
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404

    # Try to serve static files from React build directory
    try:
        if path != "" and os.path.exists(os.path.join(app.static_folder, 'react', path)):
            return send_from_directory(os.path.join(app.static_folder, 'react'), path)
        else:
            return send_from_directory(os.path.join(app.static_folder, 'react'), 'index.html')
    except:
        # If the file doesn't exist, serve React's index.html
        return send_from_directory(os.path.join(app.static_folder, 'react'), 'index.html')


# Clean up on app exit
def cleanup():
    """Cleanup function to be called when the app exits"""
    processing_queue.put(None)  # Signal worker thread to stop
    if worker_thread.is_alive():
        worker_thread.join(timeout=1.0)
    print("Cleanup complete")


# Register cleanup handler
import atexit

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True)