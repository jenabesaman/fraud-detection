import cv2
import numpy as np
from ultralytics import YOLO
import speech_recognition as sr
import threading
import queue
import os
from pathlib import Path
import time
import shutil
import signal
import sys
from datetime import datetime

try:
    import jdatetime

    PERSIAN_DATE_AVAILABLE = True
except ImportError:
    PERSIAN_DATE_AVAILABLE = False

# 🎛️ CONFIGURATION WITH DETAILED COMMENTS
CONFIG = {
    # Face Detection Settings
    'face_detection_sensitivity': 0.80,  # 🔧 MAIN PARAMETER: Controls how strict face detection is
    # 0.1 = Very strict (only clear, perfect faces detected)
    # 1.0 = Very sensitive (detects even blurry/partial faces)
    # Higher values = more faces detected but more false positives

    'face_similarity_threshold': 0.65,  # 🎯 Minimum similarity score to match a face with known person
    # 0.1 = Very loose matching (almost anyone matches)
    # 1.0 = Perfect matching required (must be identical)
    # Lower values = easier to match, higher = stricter matching

    # Object Detection Settings
    'object_confidence_threshold': 0.5,  # 📱 Minimum confidence for YOLO object detection
    # 0.1 = Detect objects even with low confidence
    # 1.0 = Only detect objects with perfect confidence
    # Higher values = fewer false object detections

    'duplicate_threshold': 0.85,  # 🔄 Similarity threshold to consider faces as duplicates
    # 0.1 = Even different faces considered duplicates
    # 1.0 = Only identical faces considered duplicates
    # Prevents saving multiple similar unknown faces

    # Audio Detection Settings
    'audio_threshold': 0.3,  # 🎤 Audio level threshold for talking detection
    # 0.1 = Very sensitive (whispers detected)
    # 1.0 = Only very loud sounds detected
    # Adjust based on room noise levels

    # Performance Settings
    'analysis_every_n_frames': 5,  # 🎬 Analyze every Nth frame for performance
    # 1 = Analyze every frame (slow but thorough)
    # 10 = Analyze every 10th frame (fast but may miss events)
    # Balance between performance and detection accuracy

    # Alert Control Settings
    'alert_cooldown_seconds': 5,  # ⏰ Minimum seconds between same type of alerts
    # Prevents spam alerts for the same issue
    # 1 = Allow alerts every second
    # 10 = Only allow alerts every 10 seconds

    'save_unknown_cooldown_seconds': 8,  # 💾 Minimum seconds between saving unknown faces
    # Prevents filling disk with duplicate unknown faces
    # Lower values = more unknown faces saved

    # Movement Detection Settings
    'look_around_sensitivity': 0.6,  # 👀 Sensitivity for detecting head movement/looking around
    # 0.1 = Only detect very large head movements
    # 1.0 = Detect even tiny head movements
    # Higher values = more sensitive to movement

    'no_face_delay_after_look_around': 15,  # 🚫 Frames to wait after look_around before checking no_face
    # Prevents false NO_FACE alerts when person looks around
    # Higher values = longer delay but fewer false positives
    # Lower values = quicker detection but more false alarms

    # Logging Settings
    'verbose_logging': False,  # 📝 Enable detailed console logging for debugging
    # True = Show detailed detection information
    # False = Only show important alerts
    # Useful for troubleshooting and fine-tuning

    'verbose_log_interval': 5,  # 📋 Seconds between verbose log messages
    # Prevents console spam when verbose logging enabled
    # Lower values = more frequent detailed logs
}

# Calculate face detection parameters based on sensitivity
FACE_DETECTION_PARAMS = {
    'face_confidence_threshold': 0.7 - (CONFIG['face_detection_sensitivity'] * 0.3),
    'face_quality_threshold': 0.8 - (CONFIG['face_detection_sensitivity'] * 0.4),
    'min_neighbors': int(6 - (CONFIG['face_detection_sensitivity'] * 3)),
    'min_face_size': (int(70 - CONFIG['face_detection_sensitivity'] * 20),
                      int(70 - CONFIG['face_detection_sensitivity'] * 20)),
    'max_face_size': (int(300 + CONFIG['face_detection_sensitivity'] * 100),
                      int(300 + CONFIG['face_detection_sensitivity'] * 100)),
    'texture_min': int(150 - CONFIG['face_detection_sensitivity'] * 100),
    'texture_max': int(5000 + CONFIG['face_detection_sensitivity'] * 5000),
    'skin_threshold': 0.4 - (CONFIG['face_detection_sensitivity'] * 0.2),
}

# Ensure parameters are within valid ranges
FACE_DETECTION_PARAMS['min_neighbors'] = max(1, FACE_DETECTION_PARAMS['min_neighbors'])
FACE_DETECTION_PARAMS['texture_min'] = max(1, FACE_DETECTION_PARAMS['texture_min'])
FACE_DETECTION_PARAMS['skin_threshold'] = max(0.1, FACE_DETECTION_PARAMS['skin_threshold'])

# Print current sensitivity settings
print(f"🎛️ Face Detection Sensitivity: {CONFIG['face_detection_sensitivity']}")
print(f"🎛️ Look Around Sensitivity: {CONFIG['look_around_sensitivity']}")
print(f"🎛️ NO_FACE Delay After Look Around: {CONFIG['no_face_delay_after_look_around']} frames")
print(f"   • Quality threshold: {FACE_DETECTION_PARAMS['face_quality_threshold']:.2f}")
print(f"   • Min neighbors: {FACE_DETECTION_PARAMS['min_neighbors']}")
print(f"   • Min face size: {FACE_DETECTION_PARAMS['min_face_size']}")


class ImprovedFaceDetector:
    """Enhanced face detection with better face vs object discrimination"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'  # Fixed: removed extra 's'
        )

        # Add validation
        if self.face_cascade.empty():
            print("⚠️ Warning: Face cascade not loaded properly")
        if self.eye_cascade.empty():
            print("⚠️ Warning: Eye cascade not loaded properly")

    def validate_face_region(self, face_roi):
        """Validate if a detected region is actually a face"""
        if face_roi is None or face_roi.size == 0:
            return False, 0.0

        h, w = face_roi.shape[:2]

        # Check aspect ratio
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.8 or aspect_ratio > 2.0:
            return False, 0.0

        # Check for eyes in the face region (only if eye cascade loaded properly)
        quality_score = 0.0

        if not self.eye_cascade.empty():
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3, minSize=(10, 10))

            has_eyes = len(eyes) >= 1
            if has_eyes:
                quality_score += 0.4
                if len(eyes) >= 2:
                    quality_score += 0.1
        else:
            # If eye detection not available, give base score
            quality_score += 0.3

        if self.has_skin_tone(face_roi):
            quality_score += 0.3

        if self.has_face_texture(face_roi):
            quality_score += 0.2

        return quality_score > FACE_DETECTION_PARAMS['face_quality_threshold'], quality_score

    def has_skin_tone(self, face_roi):
        """Check if region has skin-like colors"""
        try:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
            return skin_percentage > FACE_DETECTION_PARAMS['skin_threshold']
        except:
            return False

    def has_face_texture(self, face_roi):
        """Check if region has face-like texture patterns"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return FACE_DETECTION_PARAMS['texture_min'] < laplacian_var < FACE_DETECTION_PARAMS['texture_max']
        except:
            return True

    def detect_faces_enhanced(self, frame):
        """Enhanced face detection with validation"""
        detected_faces = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only proceed if face cascade loaded properly
        if self.face_cascade.empty():
            return detected_faces

        haar_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=FACE_DETECTION_PARAMS['min_neighbors'],
            minSize=FACE_DETECTION_PARAMS['min_face_size'],
            maxSize=FACE_DETECTION_PARAMS['max_face_size']
        )

        for (x, y, w, h) in haar_faces:
            padding = 10
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)

            face_roi = frame[y1:y2, x1:x2]
            is_valid_face, quality_score = self.validate_face_region(face_roi)

            if is_valid_face:
                detected_faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': quality_score,
                    'method': 'haar'
                })

        return self.remove_duplicate_detections(detected_faces)

    def remove_duplicate_detections(self, detections):
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        filtered = []

        for i, det1 in enumerate(detections):
            x1, y1, w1, h1 = det1['bbox']
            is_duplicate = False

            for det2 in filtered:
                x2, y2, w2, h2 = det2['bbox']

                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap

                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area

                iou = overlap_area / union_area if union_area > 0 else 0

                if iou > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(det1)

        return filtered


class SimplePersonDatabase:
    """Face database with controlled logging"""

    def __init__(self, base_dir="./persons"):
        self.base_dir = Path(base_dir)
        self.real_dir = self.base_dir / "real"
        self.unknown_dir = self.base_dir / "unknown"

        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)

        self.real_face_encodings = []
        self.real_face_names = []
        self.unknown_face_encodings = []

        self.face_detector = ImprovedFaceDetector()

        self.load_real_faces()
        self.load_existing_unknowns()

        self.last_unknown_save = 0
        self.last_verbose_log = 0

        print(
            f"👥 Database: {len(self.real_face_encodings)} real faces, {len(self.unknown_face_encodings)} existing unknowns")

    def should_log_verbose(self):
        """Check if we should show verbose logs"""
        if not CONFIG['verbose_logging']:
            return False

        current_time = time.time()
        if current_time - self.last_verbose_log > CONFIG['verbose_log_interval']:
            self.last_verbose_log = current_time
            return True
        return False

    def create_enhanced_encoding(self, face_image):
        """Create enhanced face encoding with multiple features"""
        try:
            face_resized = cv2.resize(face_image, (100, 100))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            features = []

            # Histogram features
            hist_full = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist_full = hist_full.flatten()
            hist_full = hist_full / (np.sum(hist_full) + 1e-10)
            features.extend(hist_full)

            # LBP features
            lbp_features = self.compute_lbp(gray)
            features.extend(lbp_features)

            # HOG-like features
            hog_features = self.compute_simple_hog(gray)
            features.extend(hog_features)

            # Color features
            if len(face_image.shape) == 3:
                color_features = self.compute_color_features(face_resized)
                features.extend(color_features)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return None

    def compute_lbp(self, gray_image):
        """Compute Local Binary Pattern features"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros_like(gray_image)

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    center = gray_image[i, j]
                    code = 0
                    code |= (gray_image[i - 1, j - 1] > center) << 7
                    code |= (gray_image[i - 1, j] > center) << 6
                    code |= (gray_image[i - 1, j + 1] > center) << 5
                    code |= (gray_image[i, j + 1] > center) << 4
                    code |= (gray_image[i + 1, j + 1] > center) << 3
                    code |= (gray_image[i + 1, j] > center) << 2
                    code |= (gray_image[i + 1, j - 1] > center) << 1
                    code |= (gray_image[i, j - 1] > center) << 0
                    lbp[i, j] = code

            hist = cv2.calcHist([lbp], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-10)

            return hist[:32]
        except:
            return np.zeros(32)

    def compute_simple_hog(self, gray_image):
        """Compute simplified HOG features"""
        try:
            gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=1)

            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            angle = np.arctan2(gy, gx) * 180 / np.pi
            angle[angle < 0] += 180

            hist_bins = 9
            hist = np.zeros(hist_bins)

            for i in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    bin_idx = int(angle[i, j] / 20)
                    if bin_idx >= hist_bins:
                        bin_idx = hist_bins - 1
                    hist[bin_idx] += magnitude[i, j]

            hist = hist / (np.sum(hist) + 1e-10)
            return hist
        except:
            return np.zeros(9)

    def compute_color_features(self, color_image):
        """Compute color-based features"""
        try:
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])

            hist_h = hist_h.flatten() / (np.sum(hist_h) + 1e-10)
            hist_s = hist_s.flatten() / (np.sum(hist_s) + 1e-10)

            return np.concatenate([hist_h, hist_s])
        except:
            return np.zeros(26)

    def create_simple_encoding(self, face_image):
        """Wrapper for backward compatibility"""
        return self.create_enhanced_encoding(face_image)

    def load_real_faces(self):
        """Load real faces with validation"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        real_image_files = []

        for ext in image_extensions:
            real_image_files.extend(self.real_dir.glob(ext))

        if not real_image_files:
            print(f"⚠️ No real faces found in: {self.real_dir}")
            return

        for img_file in real_image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                detected_faces = self.face_detector.detect_faces_enhanced(img)

                if detected_faces:
                    best_face = max(detected_faces, key=lambda x: x['confidence'])
                    x, y, w, h = best_face['bbox']
                    face_roi = img[y:y + h, x:x + w]
                    encoding = self.create_enhanced_encoding(face_roi)
                else:
                    encoding = self.create_enhanced_encoding(img)

                if encoding is not None:
                    self.real_face_encodings.append(encoding)
                    name = img_file.stem
                    self.real_face_names.append(name)
                    print(f"   ✅ Loaded: {name}")

            except Exception as e:
                print(f"   ❌ Error loading {img_file.name}: {e}")

    def load_existing_unknowns(self):
        """Load existing unknowns"""
        unknown_files = list(self.unknown_dir.glob("*.jpg"))

        for img_file in unknown_files[:20]:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    encoding = self.create_enhanced_encoding(img)
                    if encoding is not None:
                        self.unknown_face_encodings.append(encoding)
            except:
                pass

    def calculate_similarity(self, encoding1, encoding2):
        """Calculate similarity using multiple metrics"""
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0

            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            cosine_sim = dot_product / (norm1 * norm2)

            euclidean_dist = np.linalg.norm(encoding1 - encoding2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)

            similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim

            return float(similarity)

        except:
            return 0.0

    def find_real_person(self, face_image):
        """Check if face matches real person"""
        if len(self.real_face_encodings) == 0:
            return False, None

        face_encoding = self.create_enhanced_encoding(face_image)
        if face_encoding is None:
            return False, None

        best_similarity = 0.0
        best_match_name = None

        for i, real_encoding in enumerate(self.real_face_encodings):
            similarity = self.calculate_similarity(face_encoding, real_encoding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_name = self.real_face_names[i]

        if self.should_log_verbose():
            print(f"   🎯 Best match: {best_match_name} ({best_similarity:.3f})")

        if best_similarity > CONFIG['face_similarity_threshold']:
            return True, best_match_name
        else:
            return False, None

    def is_duplicate_unknown(self, face_image):
        """Check for duplicate unknown"""
        if len(self.unknown_face_encodings) == 0:
            return False

        face_encoding = self.create_enhanced_encoding(face_image)
        if face_encoding is None:
            return True

        for unknown_encoding in self.unknown_face_encodings:
            similarity = self.calculate_similarity(face_encoding, unknown_encoding)
            if similarity > CONFIG['duplicate_threshold']:
                return True

        return False

    def save_unknown_face(self, face_image):
        """Save unknown face with validation"""
        current_time = time.time()

        if current_time - self.last_unknown_save < CONFIG['save_unknown_cooldown_seconds']:
            return False

        is_valid_face, quality_score = self.face_detector.validate_face_region(face_image)
        if not is_valid_face:
            if self.should_log_verbose():
                print(f"   ⚠️ Not saving - low face quality score: {quality_score:.2f}")
            return False

        is_real, person_name = self.find_real_person(face_image)
        if is_real:
            return False

        if self.is_duplicate_unknown(face_image):
            return False

        try:
            timestamp = int(current_time)
            filename = f"unknown_{timestamp}_q{int(quality_score * 100)}.jpg"
            filepath = self.unknown_dir / filename

            cv2.imwrite(str(filepath), face_image)

            face_encoding = self.create_enhanced_encoding(face_image)
            if face_encoding is not None:
                self.unknown_face_encodings.append(face_encoding)
                if len(self.unknown_face_encodings) > 30:
                    self.unknown_face_encodings.pop(0)

            self.last_unknown_save = current_time
            print(f"   💾 New unknown saved: {filename} (quality: {quality_score:.2f})")
            return True

        except Exception as e:
            return False


class SimpleFraudDetector:
    def __init__(self, exam_name):
        self.exam_name = exam_name
        self.person_db = SimplePersonDatabase()
        self.setup_directories()
        self.setup_enhanced_detection()
        self.setup_yolo()
        self.setup_audio()
        self.setup_logging()

        self.frame_count = 0
        self.last_alert_time = {}
        self.detected_real_people = set()
        self.fraud_count = 0
        self.last_verbose_log = 0

        # For look around detection
        self.previous_face_positions = []
        self.face_position_history = []

        # 🆕 NEW: Add delay mechanism for NO_FACE after LOOK_AROUND
        self.last_look_around_frame = -1
        self.no_face_delay_frames = CONFIG['no_face_delay_after_look_around']

        print("✅ Fraud Detection System Ready!")

    def setup_directories(self):
        """Setup directories"""
        if PERSIAN_DATE_AVAILABLE:
            date_str = jdatetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

        self.session_dir = Path(f"./ExamLogs/{self.exam_name}-{date_str}")
        self.logs_dir = self.session_dir / "logs"
        self.saved_frames_dir = self.session_dir / "SavedFrames"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.saved_frames_dir.mkdir(exist_ok=True)

    def setup_enhanced_detection(self):
        """Setup enhanced face detection"""
        self.face_detector = ImprovedFaceDetector()
        print("✅ Enhanced face detection ready")

    def setup_yolo(self):
        """Setup YOLO for object detection"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to('cpu')

            self.person_class_id = None
            for class_id, class_name in self.yolo_model.names.items():
                if class_name == 'person':
                    self.person_class_id = class_id
                    break

            print("✅ YOLO ready for object detection")
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
            self.yolo_model = None

    def setup_audio(self):
        """Setup audio"""
        self.audio_queue = queue.Queue()
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            self.audio_thread = threading.Thread(target=self.audio_monitor, daemon=True)
            self.audio_thread.start()
            print("✅ Audio ready")
        except Exception as e:
            print(f"⚠️ Audio not available: {e}")
            self.microphone = None

    def audio_monitor(self):
        """Audio monitoring"""
        if not self.microphone:
            return

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except:
            return

        while True:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_level = float(np.max(np.abs(audio_data)) / 32768.0)
                self.audio_queue.put(audio_level)
            except:
                self.audio_queue.put(0.0)

    def setup_logging(self):
        """Setup logging"""
        try:
            self.log_file = self.logs_dir / "fraud_detection.txt"
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Fraud Detection Log ===\n")
                f.write(f"Exam Name: {self.exam_name}\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Face Detection Sensitivity: {CONFIG['face_detection_sensitivity']}\n")
                f.write(f"Look Around Sensitivity: {CONFIG['look_around_sensitivity']}\n")
                f.write(f"NO_FACE Delay After Look Around: {CONFIG['no_face_delay_after_look_around']} frames\n")
                f.write(f"Real Faces Loaded: {len(self.person_db.real_face_encodings)}\n")
                f.write("=" * 40 + "\n\n")
            print("📝 Logging ready")
        except Exception as e:
            print(f"Log error: {e}")

    def should_log_verbose(self):
        """Check if we should show verbose logs"""
        if not CONFIG['verbose_logging']:
            return False

        current_time = time.time()
        if current_time - self.last_verbose_log > CONFIG['verbose_log_interval']:
            self.last_verbose_log = current_time
            return True
        return False

    def detect_look_around(self, faces):
        """Detect if person is looking around based on face position changes"""
        if len(faces) == 0:
            return False, "No face to track"

        # Get current face position (use the largest/most confident face)
        current_face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = current_face['bbox']

        # Calculate face center
        face_center = (x + w // 2, y + h // 2)

        # Add to history
        self.face_position_history.append(face_center)

        # Keep only recent positions (last 10 frames)
        if len(self.face_position_history) > 10:
            self.face_position_history.pop(0)

        # Need at least 5 positions to detect movement
        if len(self.face_position_history) < 5:
            return False, "Building position history"

        # Calculate movement variance
        positions = np.array(self.face_position_history)
        x_variance = np.var(positions[:, 0])
        y_variance = np.var(positions[:, 1])

        # Calculate total movement
        total_variance = x_variance + y_variance

        # Threshold based on sensitivity (lower sensitivity = higher threshold)
        movement_threshold = 1000 * (1.0 - CONFIG['look_around_sensitivity'])

        if total_variance > movement_threshold:
            return True, f"Looking around detected (movement: {total_variance:.1f})"

        return False, f"Normal position (movement: {total_variance:.1f})"

    def draw_text_with_background(self, frame, text, position, font_scale=0.6, thickness=2):
        """Draw yellow text with black background"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        # Draw black background rectangle
        cv2.rectangle(frame, (x - 2, y - text_height - 5),
                      (x + text_width + 2, y + 5), (0, 0, 0), -1)

        # Draw yellow text
        cv2.putText(frame, text, position, font, font_scale, (0, 255, 255), thickness)

    def process_faces(self, faces, frame):
        """Process validated faces and count unique people"""
        detected_people = set()
        unknown_count = 0
        has_unknown = False

        if self.should_log_verbose():
            print(f"\n🔍 Processing {len(faces)} validated faces...")

        for face_data in faces:
            x, y, w, h = face_data['bbox']
            confidence = face_data['confidence']

            padding = 10
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)

            face_region = frame[y1:y2, x1:x2]

            is_real, person_name = self.person_db.find_real_person(face_region)

            if is_real:
                detected_people.add(person_name)
                self.detected_real_people.add(person_name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.draw_text_with_background(frame, f"{person_name} ({confidence:.2f})", (x, y - 10))

                if self.should_log_verbose():
                    print(f"   ✅ Real person: {person_name} (conf: {confidence:.2f})")

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                self.draw_text_with_background(frame, f"Unknown ({confidence:.2f})", (x, y - 10))

                saved = self.person_db.save_unknown_face(face_region)
                if saved:
                    unknown_count += 1
                    has_unknown = True

                if self.should_log_verbose():
                    print(f"   ❓ Unknown person (conf: {confidence:.2f})")

        total_unique_people = len(detected_people) + (1 if unknown_count > 0 else 0)

        return len(detected_people), unknown_count, total_unique_people, has_unknown

    def detect_multiple_faces(self, frame):
        """Detect faces using enhanced detection"""
        faces = self.face_detector.detect_faces_enhanced(frame)

        if len(faces) == 0:
            self.draw_text_with_background(frame, "No faces detected", (10, 30))
            return False, "No faces detected", 0, False

        real_people_count, unknown_count, total_unique_people, has_unknown = self.process_faces(faces, frame)

        status = f"Faces: {len(faces)} | People: {total_unique_people} (Real: {real_people_count}, Unknown: {unknown_count})"
        self.draw_text_with_background(frame, status, (10, 30))

        if total_unique_people > 1:
            return True, f"Multiple unique people detected: {total_unique_people}", total_unique_people, has_unknown

        return False, f"Single person detected", total_unique_people, has_unknown

    def detect_objects(self, frame):
        """Detect suspicious objects (excluding faces)"""
        if not self.yolo_model:
            return False, []

        try:
            results = self.yolo_model(frame, verbose=False)
            suspicious_objects = []
            suspicious_classes = ['cell phone', 'book', 'laptop', 'bottle', 'cup']

            face_regions = self.face_detector.detect_faces_enhanced(frame)

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]

                        if confidence < CONFIG['object_confidence_threshold']:
                            continue

                        coords = box.xyxy[0]
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                        is_face_overlap = False
                        for face_data in face_regions:
                            fx, fy, fw, fh = face_data['bbox']

                            overlap_x = max(0, min(x2, fx + fw) - max(x1, fx))
                            overlap_y = max(0, min(y2, fy + fh) - max(y1, fy))
                            overlap_area = overlap_x * overlap_y

                            obj_area = (x2 - x1) * (y2 - y1)

                            if overlap_area > 0.5 * obj_area:
                                is_face_overlap = True
                                break

                        if class_id == self.person_class_id and is_face_overlap:
                            continue

                        color = (0, 255, 255) if class_name in suspicious_classes else (255, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        self.draw_text_with_background(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10))

                        if class_name in suspicious_classes and confidence > 0.7:
                            suspicious_objects.append(f"{class_name} (conf: {confidence:.2f})")

            return len(suspicious_objects) > 0, suspicious_objects

        except Exception as e:
            return False, []

    def detect_talking(self):
        """Detect talking"""
        try:
            if not self.audio_queue.empty():
                audio_level = self.audio_queue.get_nowait()
                if audio_level > CONFIG['audio_threshold']:
                    return True, f"Talking detected (level: {audio_level:.3f})"
            return False, "Silent"
        except:
            return False, "Silent"

    def detect_no_face(self, frame):
        """Detect if no valid face is present"""
        faces = self.face_detector.detect_faces_enhanced(frame)
        if len(faces) == 0:
            return True, "No face detected - possible absence"
        return False, "Face present"

    def should_alert(self, alert_type):
        """Check if should alert"""
        current_time = time.time()
        if alert_type not in self.last_alert_time:
            self.last_alert_time[alert_type] = current_time
            return True

        if current_time - self.last_alert_time[alert_type] > CONFIG['alert_cooldown_seconds']:
            self.last_alert_time[alert_type] = current_time
            return True

        return False

    def analyze_frame(self, frame):
        """🔄 UPDATED: Analyze frame for fraud with NO_FACE delay after LOOK_AROUND"""
        alerts = []
        alert_types = []

        multi_person, multi_msg, people_count, has_unknown = self.detect_multiple_faces(frame)

        # Check for unknown face alert
        if has_unknown and self.should_alert("UNKNOWN_FACE"):
            alerts.append("Unknown face detected")
            alert_types.append("UNKNOWN_FACE")

        if multi_person and self.should_alert("MULTIPLE_PEOPLE"):
            alerts.append(multi_msg)
            alert_types.append("MULTIPLE_PEOPLE")

        # 🆕 Look around detection (check this FIRST)
        faces = self.face_detector.detect_faces_enhanced(frame)
        look_around, look_msg = self.detect_look_around(faces)
        if look_around and self.should_alert("LOOK_AROUND"):
            alerts.append(look_msg)
            alert_types.append("LOOK_AROUND")
            # Record the frame when look_around was detected
            self.last_look_around_frame = self.frame_count
            if self.should_log_verbose():
                print(
                    f"   🔄 Look around detected at frame {self.frame_count}, delaying NO_FACE for {self.no_face_delay_frames} frames")

        # 🆕 NO_FACE detection with delay after LOOK_AROUND
        frames_since_look_around = self.frame_count - self.last_look_around_frame
        should_check_no_face = frames_since_look_around > self.no_face_delay_frames

        if should_check_no_face:
            no_face, no_face_msg = self.detect_no_face(frame)
            if no_face and self.should_alert("NO_FACE"):
                alerts.append(no_face_msg)
                alert_types.append("NO_FACE")
        else:
            if self.should_log_verbose():
                remaining_frames = self.no_face_delay_frames - frames_since_look_around
                print(f"   ⏳ NO_FACE check delayed: {remaining_frames} frames remaining after look_around")

        # Rest of detection logic remains the same
        suspicious_objs, obj_list = self.detect_objects(frame)
        if suspicious_objs and self.should_alert("SUSPICIOUS_OBJECT"):
            alerts.extend(obj_list)
            alert_types.extend(["SUSPICIOUS_OBJECT"] * len(obj_list))

        talking, talk_msg = self.detect_talking()
        if talking and self.should_alert("TALKING"):
            alerts.append(talk_msg)
            alert_types.append("TALKING")

        return len(alerts) > 0, alerts, alert_types

    def save_frame(self, frame, alert_type="FRAUD"):
        """Save frame with updated naming"""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"[{alert_type}]_{date_str}.jpg"
        filepath = self.saved_frames_dir / filename

        # Create a copy for saving with yellow text and black background
        save_frame = frame.copy()

        cv2.imwrite(str(filepath), save_frame)
        self.fraud_count += 1
        return filename

    def log_event(self, alert_type, message, frame_saved=None):
        """Log event to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Frame {self.frame_count}:\n")
                f.write(f"  Alert Type: {alert_type}\n")
                f.write(f"  Message: {message}\n")
                if frame_saved:
                    f.write(f"  Saved Frame: {frame_saved}\n")
                f.write(
                    f"  Detected People: {', '.join(self.detected_real_people) if self.detected_real_people else 'None'}\n")
                f.write("-" * 40 + "\n\n")
        except Exception as e:
            print(f"Log write error: {e}")

    def cleanup(self):
        """Cleanup and write final summary"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 40 + "\n")
                f.write("SESSION SUMMARY\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Frames: {self.frame_count}\n")
                f.write(f"Total Fraud Detections: {self.fraud_count}\n")
                f.write(
                    f"Real People Detected: {', '.join(self.detected_real_people) if self.detected_real_people else 'None'}\n")
                f.write("=" * 40 + "\n")
        except:
            pass

        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


def main():
    try:
        print("🎓 Smart Fraud Detection System")
        print("=" * 35)
        exam_name = input("📝 Enter exam name: ").strip()
        if not exam_name:
            exam_name = "Test_Exam"

        verbose = input("🔧 Enable verbose mode? (y/N): ").strip().lower()
        CONFIG['verbose_logging'] = verbose == 'y'

        print(f"\n🎯 Starting detection for: {exam_name}")
        print(f"📁 Real faces folder: ./persons/real/")
        print(f"📁 Unknown faces folder: ./persons/unknown/")
        print(f"📁 Logs folder: ./ExamLogs/")

        if CONFIG['verbose_logging']:
            print(f"🔧 Verbose mode: ON (logs every {CONFIG['verbose_log_interval']}s)")
        else:
            print("🔧 Verbose mode: OFF")

        # 🆕 Show new delay configuration
        print(f"🔧 NO_FACE delay after look_around: {CONFIG['no_face_delay_after_look_around']} frames")

        detector = SimpleFraudDetector(exam_name)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n🎮 Controls:")
        print("   'q' = Quit")
        print("   's' = Save current frame")
        print("   'v' = Toggle verbose mode")
        print("\n🟢 System ready! Starting detection...")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read camera")
                break

            frame_count += 1
            detector.frame_count = frame_count

            if frame_count % CONFIG['analysis_every_n_frames'] == 0:
                is_fraud, alerts, alert_types = detector.analyze_frame(frame)

                if is_fraud:
                    print(f"\n🚨 FRAUD DETECTED - Frame {frame_count}")

                    # Save frame with the primary alert type
                    primary_alert_type = alert_types[0] if alert_types else "FRAUD"
                    saved_file = detector.save_frame(frame, primary_alert_type)

                    for i, (alert, alert_type) in enumerate(zip(alerts, alert_types)):
                        print(f"   {i + 1}. [{alert_type}] {alert}")
                        detector.log_event(alert_type, alert, saved_file if i == 0 else None)

                    print(f"   💾 Saved: {saved_file}")

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            detector.draw_text_with_background(frame, f"Exam: {exam_name}", (10, frame.shape[0] - 100))
            detector.draw_text_with_background(frame, f"Frame: {frame_count} | FPS: {fps:.1f}",
                                               (10, frame.shape[0] - 80))
            detector.draw_text_with_background(frame, f"Fraud Detections: {detector.fraud_count}",
                                               (10, frame.shape[0] - 60))
            detector.draw_text_with_background(frame, f"Look Around Sensitivity: {CONFIG['look_around_sensitivity']}",
                                               (10, frame.shape[0] - 40))
            # 🆕 Show delay status
            frames_since_look_around = frame_count - detector.last_look_around_frame
            if frames_since_look_around <= detector.no_face_delay_frames:
                remaining = detector.no_face_delay_frames - frames_since_look_around
                detector.draw_text_with_background(frame, f"NO_FACE delayed: {remaining} frames",
                                                   (10, frame.shape[0] - 20))

            cv2.imshow(f'Fraud Detection - {exam_name}', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                saved_file = detector.save_frame(frame, "MANUAL_SAVE")
                print(f"💾 Manual save: {saved_file}")
                detector.log_event("MANUAL_SAVE", "Manual frame save", saved_file)
            elif key == ord('v'):
                CONFIG['verbose_logging'] = not CONFIG['verbose_logging']
                print(f"🔧 Verbose mode: {'ON' if CONFIG['verbose_logging'] else 'OFF'}")

        cap.release()
        detector.cleanup()

        print(f"\n📊 Session Summary:")
        print(f"   • Exam: {exam_name}")
        print(f"   • Total frames: {frame_count}")
        print(f"   • Average FPS: {fps:.1f}")
        print(f"   • Fraud detections: {detector.fraud_count}")
        print(f"   • Real people detected: {len(detector.detected_real_people)}")
        for person in detector.detected_real_people:
            print(f"     - {person}")
        print(f"   • Session folder: {detector.session_dir}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()