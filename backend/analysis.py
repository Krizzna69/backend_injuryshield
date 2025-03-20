import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib

# Use Agg backend to avoid display issues on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
import gc  # Garbage collector
import time
from threading import Thread

warnings.filterwarnings('ignore')

# Initialize MediaPipe Pose with lower complexity for server environments
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def reduce_video_size(input_path, output_path, target_size_mb=30):
    """Reduce video size before processing to save memory"""
    import os
    import subprocess

    # Calculate current size in MB
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)

    if file_size_mb <= target_size_mb:
        # If file is already small enough, just copy it
        import shutil
        shutil.copyfile(input_path, output_path)
        return output_path

    # Calculate target bitrate (approximation)
    try:
        from subprocess import check_output

        # Get video duration using ffprobe
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
        try:
            duration = float(check_output(cmd, shell=True).decode('utf-8').strip())
        except:
            # If ffprobe fails, estimate duration from OpenCV
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 60
            cap.release()

        # Calculate target bitrate in kbps (more aggressive reduction)
        target_bitrate_kbps = int((target_size_mb * 8 * 1024) / duration * 0.8)  # 20% buffer

        # Use ffmpeg to reduce video size
        cmd = (
            f'ffmpeg -i "{input_path}" -b:v {target_bitrate_kbps}k -bufsize {target_bitrate_kbps * 2}k '
            f'-maxrate {target_bitrate_kbps * 1.5}k -vf "scale=640:-2" -r 15 -y "{output_path}"'
        )

        subprocess.call(cmd, shell=True)
        return output_path

    except Exception as e:
        print(f"Error reducing video size: {e}, using original video")
        import shutil
        shutil.copyfile(input_path, output_path)
        return output_path


def process_video(video_path, max_frames=300):
    """Process video and extract pose landmarks with memory optimization"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create lists to store landmarks
    all_landmarks = []
    frame_indices = []

    # Process video at lower framerate for efficiency and max frames for memory
    sample_rate = max(1, int(fps / 5))  # Process ~5 frames per second
    total_frames_to_process = min(frame_count, sample_rate * max_frames)

    # Initialize pose with lower complexity for server use
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Lower complexity to use less memory
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        current_frame = 0
        processed = 0
        last_gc_time = time.time()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Only process every nth frame and limit total frames
            if current_frame % sample_rate == 0 and processed < total_frames_to_process:
                # Resize image to reduce memory usage
                image = cv2.resize(image, (640, 480))

                # Convert to RGB and process
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # Clear memory to avoid leaks
                del image_rgb

                # If landmarks detected, save them
                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    all_landmarks.append(landmarks)
                    frame_indices.append(current_frame)
                    processed += 1

                # Force garbage collection every 10 seconds
                if time.time() - last_gc_time > 10:
                    gc.collect()
                    last_gc_time = time.time()

                # Print progress less frequently to reduce overhead
                if processed % 10 == 0:
                    print(f"Processed {processed} frames out of target {total_frames_to_process}")

            current_frame += 1

            # Break early if we've processed enough frames
            if processed >= total_frames_to_process:
                break

    cap.release()
    gc.collect()  # Force garbage collection

    return all_landmarks, frame_indices, fps


def create_visualization(video_path, frame_indices, risk_scores, output_dir, quality='low'):
    """Create a visualization with optimized memory usage"""
    # Create a reduced size video first
    reduced_video_path = os.path.join(output_dir, 'reduced_input.mp4')
    try:
        video_path = reduce_video_size(video_path, reduced_video_path, target_size_mb=20)
    except Exception as e:
        print(f"Error reducing video size: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Set fixed output size based on quality
    if quality == 'low':
        width, height = 480, 360
        fps = 10
    elif quality == 'medium':
        width, height = 640, 480
        fps = 15
    else:
        width, height = 854, 480
        fps = 20

    # Define output path
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'analyzed_video.mp4')

    # Try different codecs
    codecs = [('avc1', cv2.VideoWriter_fourcc(*'avc1')),
              ('X264', cv2.VideoWriter_fourcc(*'X264')),
              ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))]

    out = None
    for codec_name, codec in codecs:
        try:
            out = cv2.VideoWriter(out_path, codec, fps, (width, height))
            if out.isOpened():
                print(f"Using {codec_name} codec for video")
                break
        except Exception as e:
            print(f"Could not use {codec_name} codec: {e}")

    if not out or not out.isOpened():
        raise ValueError("Could not create video writer with any codec")

    # Create mapping from frame index to risk score
    risk_map = {idx: score for idx, score in zip(frame_indices, risk_scores)}

    # Process fewer frames for output video
    frame_skip = max(3, int(cap.get(cv2.CAP_PROP_FPS) / fps))

    # Use lower complexity pose model for visualization
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lowest complexity for visualization
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        current_frame = 0
        frames_written = 0
        max_frames = 300  # Limit output video frames

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Skip frames to reduce file size
            if current_frame % frame_skip != 0:
                current_frame += 1
                continue

            # Resize image for reduced file size
            image = cv2.resize(image, (width, height))

            # If current frame is one we analyzed
            if current_frame in risk_map:
                risk = risk_map[current_frame]

                # Process image for pose landmarks (use RGB to save memory)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                del image_rgb  # Free memory

                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )

                # Draw risk score
                if risk < 0.3:
                    color = (0, 255, 0)  # Green (low risk)
                    risk_text = "Low Injury Risk"
                elif risk < 0.7:
                    color = (0, 255, 255)  # Yellow (medium risk)
                    risk_text = "Moderate Injury Risk"
                else:
                    color = (0, 0, 255)  # Red (high risk)
                    risk_text = "High Injury Risk"

                # Add text to image
                cv2.putText(
                    image,
                    f"{risk_text}: {risk:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Smaller font
                    color,
                    2
                )

            # Write frame to output video
            out.write(image)
            frames_written += 1
            current_frame += 1

            # Progress updates less frequently
            if frames_written % 20 == 0:
                print(f"Writing frame {frames_written}")

            # Force garbage collection periodically
            if frames_written % 50 == 0:
                gc.collect()

            # Limit total frames in output video
            if frames_written >= max_frames:
                break

    cap.release()
    out.release()
    print(f"Analyzed video saved to: {out_path}")

    # Clean up temp file
    if os.path.exists(reduced_video_path) and reduced_video_path != video_path:
        try:
            os.remove(reduced_video_path)
        except:
            pass

    return out_path


def analyze_running_form(video_path, output_dir, quality='low'):
    """Main function with memory optimizations"""
    # Always reduce video size first to prevent memory issues
    reduced_video_path = os.path.join(output_dir, 'reduced_input.mp4')
    try:
        os.makedirs(output_dir, exist_ok=True)
        video_path = reduce_video_size(video_path, reduced_video_path, target_size_mb=25)
        print(f"Reduced video size, now working with: {video_path}")
    except Exception as e:
        print(f"Error reducing video: {e}, continuing with original")

    try:
        print("Processing video...")
        landmarks_data, frame_indices, fps = process_video(video_path)

        if not landmarks_data:
            raise ValueError("No pose landmarks detected in video")

        print("Extracting biomechanical features...")
        features = extract_running_features(landmarks_data)

        # Free memory
        del landmarks_data
        gc.collect()

        print("Assessing injury risk...")
        risk_scores = assess_injury_risk(features)

        # Identify risk factors and generate recommendations
        risk_factors = identify_risk_factors(features)
        recommendations = generate_recommendations(risk_factors)

        # Free more memory
        del features
        gc.collect()

        print("Creating visualization...")
        visualization_path = create_visualization(video_path, frame_indices, risk_scores, output_dir, quality)

        # Calculate summary statistics
        avg_risk = float(np.mean(risk_scores))
        peak_risk = float(np.max(risk_scores))

        result = {
            'average_risk': avg_risk,
            'peak_risk': peak_risk,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'visualization_path': visualization_path
        }

        # Save the plot in a separate thread to avoid blocking
        thread = Thread(target=save_risk_plot, args=(risk_scores, output_dir))
        thread.start()

        # Save report
        save_report(result, output_dir)

        # Clean up temporary files
        try:
            if os.path.exists(reduced_video_path) and reduced_video_path != video_path:
                os.remove(reduced_video_path)
        except:
            pass

        return result, risk_scores, frame_indices

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during analysis: {e}")

        # Return minimal results if we failed
        return {
            'average_risk': 0.5,
            'peak_risk': 0.7,
            'risk_factors': {'Processing Error': True},
            'recommendations': [
                "Video processing encountered an error. Try uploading a clearer video with a person running visible in the frame."],
            'visualization_path': video_path
        }, [0.5], [0]


# The rest of your functions (calculate_angle, extract_running_features, etc.) can remain the same
# I'll keep them here for completeness

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def extract_running_features(landmarks_data):
    """Extract biomechanical features relevant to running injuries"""
    features = []

    for frame_landmarks in landmarks_data:
        # Convert flat list to dictionary for easier access
        landmark_dict = {}
        for i in range(0, len(frame_landmarks), 4):
            idx = i // 4
            landmark_dict[idx] = {
                'x': frame_landmarks[i],
                'y': frame_landmarks[i + 1],
                'z': frame_landmarks[i + 2],
                'visibility': frame_landmarks[i + 3]
            }

        # Extract key points
        left_hip = landmark_dict[23]
        right_hip = landmark_dict[24]
        left_knee = landmark_dict[25]
        right_knee = landmark_dict[26]
        left_ankle = landmark_dict[27]
        right_ankle = landmark_dict[28]
        left_foot = landmark_dict[31]
        right_foot = landmark_dict[32]

        # Calculate key angles (in degrees)
        left_knee_angle = calculate_angle(
            [left_hip['x'], left_hip['y']],
            [left_knee['x'], left_knee['y']],
            [left_ankle['x'], left_ankle['y']]
        )

        right_knee_angle = calculate_angle(
            [right_hip['x'], right_hip['y']],
            [right_knee['x'], right_knee['y']],
            [right_ankle['x'], right_ankle['y']]
        )

        # Calculate stride metrics
        stride_length = abs(left_foot['x'] - right_foot['x'])
        stride_width = abs(left_foot['z'] - right_foot['z'])

        # Hip drop (pelvic tilt)
        hip_drop = abs(left_hip['y'] - right_hip['y'])

        # Foot strike pattern (lower value suggests more forefoot strike)
        left_foot_strike = left_foot['y'] - left_ankle['y']
        right_foot_strike = right_foot['y'] - right_ankle['y']

        # Compile features
        frame_features = [
            left_knee_angle,
            right_knee_angle,
            stride_length,
            stride_width,
            hip_drop,
            left_foot_strike,
            right_foot_strike
        ]

        features.append(frame_features)

    return features


def assess_injury_risk(features):
    """Assess injury risk based on biomechanical features"""
    risk_scores = []

    # Define risk thresholds based on biomechanical literature
    knee_valgus_threshold = 170  # Degrees (less is higher risk)
    stride_width_threshold = 0.2  # Normalized (more is higher risk)
    hip_drop_threshold = 0.05  # Normalized (more is higher risk)
    foot_strike_threshold = 0.05  # Normalized (less can be more risk - forefoot running without proper training)

    for frame in features:
        left_knee_angle, right_knee_angle, stride_length, stride_width, hip_drop, left_foot_strike, right_foot_strike = frame

        # Calculate individual risk factors (0-1 scale, higher is more risky)
        knee_risk = (2 - (left_knee_angle / knee_valgus_threshold) - (right_knee_angle / knee_valgus_threshold)) / 2
        knee_risk = np.clip(knee_risk, 0, 1)

        stride_width_risk = min(stride_width / stride_width_threshold, 1)
        hip_drop_risk = min(hip_drop / hip_drop_threshold, 1)

        # Both very high and very low foot strike can be risky
        left_foot_strike_risk = min(abs(left_foot_strike - foot_strike_threshold) / foot_strike_threshold, 1)
        right_foot_strike_risk = min(abs(right_foot_strike - foot_strike_threshold) / foot_strike_threshold, 1)
        foot_strike_risk = (left_foot_strike_risk + right_foot_strike_risk) / 2

        # Overall risk score (weighted average)
        risk_score = (
                0.40 * knee_risk +  # Knee angles are strong injury predictors
                0.20 * stride_width_risk +  # Stride width affects lateral stability
                0.25 * hip_drop_risk +  # Hip drop affects loading patterns
                0.15 * foot_strike_risk  # Foot strike affects impact forces
        )

        risk_scores.append(risk_score)

    return risk_scores


def identify_risk_factors(features):
    """Identify specific risk factors from running form"""
    risk_factors = {
        'Knee Valgus': False,
        'Excessive Hip Drop': False,
        'Improper Stride Width': False,
        'Suboptimal Foot Strike': False
    }

    avg_features = np.mean(features, axis=0)
    left_knee_angle, right_knee_angle, stride_length, stride_width, hip_drop, left_foot_strike, right_foot_strike = avg_features

    # Check for knee valgus (knees collapsing inward)
    if left_knee_angle < 170 or right_knee_angle < 170:
        risk_factors['Knee Valgus'] = True

    # Check for excessive hip drop
    if hip_drop > 0.05:
        risk_factors['Excessive Hip Drop'] = True

    # Check for improper stride width
    if stride_width > 0.2 or stride_width < 0.05:
        risk_factors['Improper Stride Width'] = True

    # Check foot strike pattern
    if left_foot_strike < 0.01 or right_foot_strike < 0.01:
        risk_factors['Suboptimal Foot Strike'] = True

    return risk_factors


def generate_recommendations(risk_factors):
    """Generate comprehensive training recommendations based on identified risk factors"""
    recommendations = []

    if risk_factors.get('Knee Valgus'):
        recommendations.append(
            "Strengthen hip abductors (gluteus medius) with exercises like clamshells, side leg raises, and band walks")
        recommendations.append("Focus on knee alignment during squats and running drills")

    if risk_factors.get('Excessive Hip Drop'):
        recommendations.append(
            "Incorporate single-leg stability exercises like single-leg deadlifts and Bulgarian split squats")
        recommendations.append("Strengthen core and hip stabilizers with planks and hip bridges")
        recommendations.append("Add lateral band walks to activate hip stabilizers before running")

    if risk_factors.get('Improper Stride Width'):
        recommendations.append("Work on running form drills focusing on proper foot placement")
        recommendations.append("Consider gait analysis with a physical therapist")
        recommendations.append("Practice running between lines to calibrate optimal stride width")

    if risk_factors.get('Suboptimal Foot Strike'):
        recommendations.append("Gradual transition to mid-foot striking with shorter strides")
        recommendations.append("Strengthen foot and ankle muscles with toe curls and ankle exercises")
        recommendations.append("Consider footwear with appropriate support for your running style")

    if not recommendations:
        recommendations.append(
            "Continue current training regimen with focus on gradual progression (increase volume by no more than 10% weekly)")
        recommendations.append(
            "Maintain good strength training habits with focus on posterior chain and core for injury prevention")

    return recommendations[:5]  # Limit to 5 recommendations to save memory


def save_risk_plot(risk_scores, output_dir):
    """Save risk over time plot with memory optimization"""
    plt.figure(figsize=(8, 4))  # Smaller figure size
    plt.plot([i / len(risk_scores) for i in range(len(risk_scores))], risk_scores, 'b-', linewidth=2)
    plt.axhline(y=0.3, color='g', linestyle='--', label='Low Risk')
    plt.axhline(y=0.7, color='r', linestyle='--', label='High Risk')
    plt.fill_between([i / len(risk_scores) for i in range(len(risk_scores))], risk_scores,
                     color='blue', alpha=0.2)
    plt.title('Running Injury Risk Over Time')
    plt.xlabel('Time')
    plt.ylabel('Risk Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot with lower DPI
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'risk_over_time.png')
    plt.savefig(plot_path, dpi=80)  # Lower DPI for smaller file size
    plt.close()  # Close to free memory

    return plot_path


def save_report(result, output_dir):
    """Save analysis results to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    risk_level = "Low" if result['average_risk'] < 0.3 else "Moderate" if result['average_risk'] < 0.7 else "High"
    peak_risk_level = "Low" if result['peak_risk'] < 0.3 else "Moderate" if result['peak_risk'] < 0.7 else "High"

    with open(report_path, 'w') as f:
        f.write(f"===== RUNNING INJURY RISK ANALYSIS =====\n")
        f.write(f"Overall Risk Level: {risk_level} ({result['average_risk']:.2f})\n")
        f.write(f"Peak Risk Level: {peak_risk_level} ({result['peak_risk']:.2f})\n")
        f.write("\nIdentified Risk Factors:\n")
        for factor, present in result['risk_factors'].items():
            if present:
                f.write(f"- {factor}\n")

        f.write("\nRecommendations:\n")
        for i, rec in enumerate(result['recommendations'], 1):
            f.write(f"{i}. {rec}\n")

    return report_path