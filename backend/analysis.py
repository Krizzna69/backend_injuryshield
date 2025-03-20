import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib

# Use Agg backend to avoid display issues on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def process_video(video_path):
    """Process video and extract pose landmarks"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create lists to store landmarks
    all_landmarks = []
    frame_indices = []

    # Process video at lower framerate for efficiency
    sample_rate = max(1, int(fps / 10))  # Process ~10 frames per second

    current_frame = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Only process every nth frame
        if current_frame % sample_rate == 0:
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find pose landmarks
            results = pose.process(image_rgb)

            # If landmarks detected, save them
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                all_landmarks.append(landmarks)
                frame_indices.append(current_frame)

        current_frame += 1

        # Show progress
        if current_frame % 30 == 0:
            print(f"Processing frame {current_frame}/{frame_count}")

    cap.release()

    return all_landmarks, frame_indices, fps


def create_visualization(video_path, frame_indices, risk_scores, output_dir, quality='medium'):
    """Create a visualization with pose landmarks and risk scores using browser-compatible codec"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reduce resolution based on quality setting
    if quality == 'low':
        # Reduce to 480p or smaller
        new_height = min(480, height)
        new_width = int(width * (new_height / height))
        width, height = new_width, new_height
        # Also reduce framerate
        fps = min(15, fps)
    elif quality == 'medium':
        # Reduce to 720p or smaller
        new_height = min(720, height)
        new_width = int(width * (new_height / height))
        width, height = new_width, new_height
    # 'high' quality keeps original resolution

    # Define the codec and create VideoWriter object
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'analyzed_video.mp4')

    # Try different codecs for maximum browser compatibility
    codec_success = False

    # First try with h264/avc1 which has better browser compatibility
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if out.isOpened():
            codec_success = True
            print("Using avc1 codec for video")
    except Exception as e:
        print(f"Could not use avc1 codec: {e}")

    # If that fails, try x264
    if not codec_success:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if out.isOpened():
                codec_success = True
                print("Using X264 codec for video")
        except Exception as e:
            print(f"Could not use X264 codec: {e}")

    # If that fails too, fall back to mp4v
    if not codec_success:
        print("Falling back to mp4v codec")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Could not create video writer with any codec")

    # Create mapping from frame index to risk score
    risk_map = {idx: score for idx, score in zip(frame_indices, risk_scores)}

    current_frame = 0
    # Skip frames to reduce file size
    frame_skip = 1
    if quality == 'low':
        frame_skip = 3  # Process only every 3rd frame
    elif quality == 'medium':
        frame_skip = 2  # Process every other frame

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Reduced complexity for faster processing
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Skip frames to reduce file size
            if current_frame % frame_skip != 0:
                current_frame += 1
                continue

            # Resize image for reduced file size
            if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                image = cv2.resize(image, (width, height))

            # If current frame is one we analyzed
            if current_frame in risk_map:
                risk = risk_map[current_frame]

                # Process image for pose landmarks
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

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
                    1,
                    color,
                    2
                )

            # Write frame to output video
            out.write(image)
            current_frame += 1

            # Show progress
            if current_frame % 30 == 0:
                print(f"Processing visualization frame {current_frame}/{frame_count}")

    cap.release()
    out.release()
    print(f"Analyzed video saved to: {out_path}")

    # Double check file exists and has content
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path)
        print(f"Video file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        if file_size < 1000:
            print("Warning: Video file is very small, may be corrupted")
    else:
        print("Warning: Video file was not created successfully")

    # If file is still too large, try additional compression
    max_size_mb = 25  # Maximum file size in MB
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        print(
            f"File size ({file_size_mb:.2f} MB) exceeds target size ({max_size_mb} MB). Attempting further compression...")
        try:
            compressed_path = compress_video(out_path, output_dir)
            if compressed_path:
                print(f"Using compressed version: {compressed_path}")
                return compressed_path
        except Exception as e:
            print(f"Compression failed: {e}, using original file")

    return out_path


def compress_video(input_path, output_dir):
    """Compress video file using FFmpeg if available"""
    try:
        import subprocess
        compressed_path = os.path.join(output_dir, 'compressed_video.mp4')

        # Try to use FFmpeg for better compression
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', '28',  # Higher CRF = more compression (normal range: 18-28)
            '-preset', 'fast',  # Fast encoding
            '-vf', 'scale=-2:480',  # Resize to 480p height
            '-r', '15',  # 15 fps
            '-y',  # Overwrite output file
            compressed_path
        ]

        subprocess.run(cmd, check=True)

        # Check if compression succeeded and reduced file size
        if os.path.exists(compressed_path):
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(compressed_path)

            print(f"Original: {original_size / 1024 / 1024:.2f} MB, Compressed: {compressed_size / 1024 / 1024:.2f} MB")

            if compressed_size < original_size:
                return compressed_path
    except Exception as e:
        print(f"FFmpeg compression failed: {e}")

    return None


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
        # MediaPipe landmarks reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        left_hip = landmark_dict[23]
        right_hip = landmark_dict[24]
        left_knee = landmark_dict[25]
        right_knee = landmark_dict[26]
        left_ankle = landmark_dict[27]
        right_ankle = landmark_dict[28]
        left_foot = landmark_dict[31]
        right_foot = landmark_dict[32]

        # Calculate key angles (in degrees)

        # Left knee angle
        left_knee_angle = calculate_angle(
            [left_hip['x'], left_hip['y']],
            [left_knee['x'], left_knee['y']],
            [left_ankle['x'], left_ankle['y']]
        )

        # Right knee angle
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
    """
    Assess injury risk based on biomechanical features
    Using rules based on sports medicine literature
    """
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

    if risk_factors.get('Excessive Hip Drop'):
        recommendations.append(
            "Incorporate single-leg stability exercises like single-leg deadlifts and Bulgarian split squats")
        recommendations.append("Strengthen core and hip stabilizers with planks and hip bridges")
        recommendations.append("Add lateral band walks and monster walks to your warm-up routine to activate glutes")
        recommendations.append(
            "Practice side plank variations with hip abduction to target obliques and lateral hip stabilizers")
        recommendations.append("Utilize mirror feedback during running to maintain level pelvis position")

    if risk_factors.get('Improper Stride Width'):
        recommendations.append("Work on running form drills focusing on proper foot placement")
        recommendations.append("Consider gait analysis with a physical therapist")
        recommendations.append("Practice running between lines on a track to calibrate optimal stride width")
        recommendations.append("Use ladder drills to improve foot placement accuracy and neuromuscular control")
        recommendations.append("Implement cadence training with a metronome to optimize stride mechanics")

    if risk_factors.get('Suboptimal Foot Strike'):
        recommendations.append("Gradual transition to mid-foot striking with shorter strides")
        recommendations.append("Strengthen foot and ankle muscles with toe curls and ankle stability exercises")
        recommendations.append("Use barefoot running drills on grass to improve foot proprioception")
        recommendations.append("Consider footwear with appropriate support for your foot type and running style")
        recommendations.append(
            "Practice downhill running with focus on controlled foot landing to reduce impact forces")

    # New risk factors with recommendations
    if risk_factors.get('Poor Core Engagement'):
        recommendations.append("Incorporate plank variations and anti-rotation exercises into your strength routine")
        recommendations.append("Practice diaphragmatic breathing during both strength work and running")
        recommendations.append("Add Pallof press exercises to improve core stability during rotational movements")
        recommendations.append("Use feedback cues like 'tall spine' during running to maintain proper posture")

    if risk_factors.get('Limited Ankle Mobility'):
        recommendations.append("Perform daily ankle mobility exercises including weighted and unweighted calf raises")
        recommendations.append("Use self-myofascial release techniques on calves and plantar fascia")
        recommendations.append("Incorporate eccentric heel drops to strengthen the Achilles tendon complex")
        recommendations.append("Practice balance exercises on unstable surfaces to improve ankle proprioception")

    if risk_factors.get('Excessive Forward Lean'):
        recommendations.append("Strengthen posterior chain muscles with deadlifts and back extensions")
        recommendations.append(
            "Practice running tall with cues to maintain a slight forward lean from ankles, not hips")
        recommendations.append("Improve thoracic spine mobility with foam roller extension exercises")
        recommendations.append("Add face pulls and band pull-aparts to strengthen upper back and improve posture")

    if risk_factors.get('Overstriding'):
        recommendations.append("Increase cadence by 5-10% using a metronome app during training runs")
        recommendations.append("Practice quick feet drills and high knees to reinforce faster turnover")
        recommendations.append("Focus on pulling the foot up quickly rather than reaching forward with each stride")
        recommendations.append("Use downhill running drills to practice controlled, shorter strides")

    if risk_factors.get('Muscle Imbalance'):
        recommendations.append(
            "Perform a functional movement screen with a physical therapist to identify specific imbalances")
        recommendations.append("Incorporate unilateral exercises to address strength differences between sides")
        recommendations.append("Add mobility exercises targeting tight muscles identified during assessment")
        recommendations.append("Implement progressive loading of underactive muscles through isolated exercises")

    if not recommendations:
        recommendations.append(
            "Continue current training regimen with focus on gradual progression (increase volume by no more than 10% weekly)")
        recommendations.append(
            "Maintain good strength training habits with focus on posterior chain and core for injury prevention")
        recommendations.append(
            "Implement regular recovery protocols including proper nutrition, sleep, and soft tissue maintenance")
        recommendations.append(
            "Consider periodic biomechanical assessments to catch potential issues before they lead to injury")
        recommendations.append(
            "Integrate cross-training activities to develop well-rounded fitness and reduce repetitive stress")

    return recommendations


def analyze_running_form(video_path, output_dir, quality='medium'):
    """Main function to analyze running form and predict injury risks"""
    print("Processing video...")
    landmarks_data, frame_indices, fps = process_video(video_path)

    print("Extracting biomechanical features...")
    features = extract_running_features(landmarks_data)

    print("Assessing injury risk...")
    risk_scores = assess_injury_risk(features)

    # Identify risk factors and generate recommendations
    risk_factors = identify_risk_factors(features)
    recommendations = generate_recommendations(risk_factors)

    print("Creating visualization...")
    visualization_path = create_visualization(video_path, frame_indices, risk_scores, output_dir, quality)

    # Calculate summary statistics
    avg_risk = np.mean(risk_scores)
    peak_risk = np.max(risk_scores)

    result = {
        'average_risk': avg_risk,
        'peak_risk': peak_risk,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'visualization_path': visualization_path
    }

    return result, risk_scores, frame_indices


def save_risk_plot(risk_scores, output_dir):
    """Save risk over time plot to a file"""
    plt.figure(figsize=(12, 6))
    plt.plot([i / len(risk_scores) for i in range(len(risk_scores))], risk_scores, 'b-', linewidth=2)
    plt.axhline(y=0.3, color='g', linestyle='--', label='Low Risk Threshold')
    plt.axhline(y=0.7, color='r', linestyle='--', label='High Risk Threshold')
    plt.fill_between([i / len(risk_scores) for i in range(len(risk_scores))], risk_scores,
                     color='blue', alpha=0.2)
    plt.title('Running Injury Risk Over Time', fontsize=16)
    plt.xlabel('Normalized Time', fontsize=14)
    plt.ylabel('Injury Risk Score', fontsize=14)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'risk_over_time.png')
    plt.savefig(plot_path)
    print(f"Risk plot saved to: {plot_path}")

    # Close the plot to avoid memory issues
    plt.close()

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

        f.write(f"\nAnalyzed video: {result['visualization_path']}\n")

    print(f"Analysis report saved to: {report_path}")
    return report_path