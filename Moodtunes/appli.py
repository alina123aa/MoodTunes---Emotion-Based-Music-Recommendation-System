import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import base64
import tempfile
import logging
import traceback
from flask_dance.contrib.google import make_google_blueprint, google
from pydub import AudioSegment
import random
from datetime import timedelta
from flask_session import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alina123456789secretkey'
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True

# Initialize Flask-Session
Session(app)

# Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'moodtunes'
app.config['MYSQL_PASSWORD'] = 'alina'
app.config['MYSQL_DB'] = 'moodtunes'

mysql = MySQL(app)
CORS(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Google OAuth Configuration
GOOGLE_CLIENT_ID = "293853184931-fptjkke22fesaceuen17kbc2d56fiueb.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"

google_bp = make_google_blueprint(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]
)
app.register_blueprint(google_bp, url_prefix="/login")

# Store all YouTube Music Playlists
playlists = {
    "neutral": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37uk_tjHNIsoRHSbNaP0mwu-&si=WELMDxrLGjF0-F1g",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37vM0Zg2L6_JZ94LhXzl5Bd3&si=vEzZc-E5X2FLTpJO",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37uNGP60aBFQohdjBWTbJhAW&si=PO_a6bCyysE04GwG"
    ],
    "happy": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37uMhId3l6FVXuntPja1N_KQ&si=iVjml2KfmYvgqqwq",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37sqOmvkrKqKJRyPNAzoyQLA&si=gSzhIXETK2W-E6OD",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37txkKa7jSH-8eMdOrGpOZMa&si=L5CgIxFvriv5gDgc"
    ],
    "sad": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37tmIr1QSaGDnByR2_yWHvtk&si=sdRRwJL-fzzRliZP",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37vsycG1M3vAa8yqAa1Ay_Qv&si=kAYLMJAPWP_YI87L",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37vQ_pknSe2gV_QTFjKXajQt&si=wvHkyvfBgpjSer91"
    ],
    "angry": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37vDTyudI5pM1f5JFxJAGgyA&si=i56vtAOAKDbRpaOg",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37sftv_tr6hBBtDTiV4Dd1M3&si=nKkrIvBc1Edd67sJ",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37uCZajoDTg5Q22TgNN9vq0F&si=4CjaUmQHsGCjgtOS"
    ],
    "fearful": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37ukU5rbvmSjI009eQMJ3ybZ&si=bAZUgqzpj3Nj7-ei",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37tH0nuko2Mq2t3-RrX7wtLJ&si=fXO7ug15IgTLodXD",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37uPMlFzqGkp0VHW-vK0NycK&si=bjLoyaSmIMCUh144"
    ],
    "surprised": [
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37tb8Da4kMiWIqEnOv4L9fFe&si=H30KtlX0OyiuG7Tt",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37swfNhL4zwF_LxqQ-ewV942&si=Sgx2Q6Usiiub6NF0",
        "https://music.youtube.com/playlist?list=PLpZgDvLZI37vgMHyy5uI5BZys_JBcR8WS&si=VoJVWJKExia6BZ7F"
    ]
}

# Model Paths and Initialization
MODEL_DIR = r'C:\Users\alina_2pzmzug\moodtunes\models'
FACE_MODEL_PATH = os.path.join(MODEL_DIR, 'facemodel.h5')
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, 'voicemodel.h5')

# Global variables for models
facial_model = None
voice_model = None
face_cascade = None

# Create a safer folder to store audio files
TEMP_AUDIO_PATH = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)

# Path to emotion-based songs folder
SONGS_DIR = r"C:\Users\alina_2pzmzug\moodtunes\songs"

def calculate_ensemble_emotion(facial_emotion, voice_emotion, facial_confidence, voice_confidence):
    """
    Calculate final emotion using confidence-based dynamic weighting.
    More confident predictions get higher influence.
    Ensures proper formatting of confidence values (0-100% range).
    """
    facial_emotion = facial_emotion.lower()
    voice_emotion = voice_emotion.lower()
    
    # Ensure confidence values are within 0 to 1
    facial_confidence = max(0.0, min(facial_confidence, 1.0))
    voice_confidence = max(0.0, min(voice_confidence, 1.0))

    # If both emotions are the same, return it with averaged confidence
    if facial_emotion == voice_emotion:
        final_confidence = (facial_confidence + voice_confidence) / 2
        return facial_emotion, round(final_confidence * 100, 1)  # Scaling to percentage

    # Compute dynamic weights based on confidence scores
    total_confidence = facial_confidence + voice_confidence
    if total_confidence == 0:
        return "neutral", 50.0  # Default case if confidence values are zero

    facial_weight = facial_confidence / total_confidence
    voice_weight = voice_confidence / total_confidence

    # Compute final confidence **correctly**
    final_confidence = (facial_confidence * facial_weight) + (voice_confidence * voice_weight)

    # Scale final confidence to percentage and ensure it's within 0-100%
    final_confidence = round(final_confidence * 100, 1)
    
    # Choose the emotion with the higher weighted confidence
    if facial_weight >= voice_weight:
        return facial_emotion, final_confidence
    else:
        return voice_emotion, final_confidence


def load_models():
    """Load all required models with error handling."""
    global facial_model, voice_model, face_cascade
    
    try:
        if os.path.exists(FACE_MODEL_PATH):
            facial_model = load_model(FACE_MODEL_PATH)
            logger.info("✅ Facial model loaded successfully")
        else:
            logger.error(f"❌ Facial model not found at {FACE_MODEL_PATH}")

        if os.path.exists(VOICE_MODEL_PATH):
            voice_model = load_model(VOICE_MODEL_PATH)
            logger.info("✅ Voice model loaded successfully")
            logger.info(f"Voice model input shape: {voice_model.input_shape}")
        else:
            logger.error(f"❌ Voice model not found at {VOICE_MODEL_PATH}")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("✅ Face cascade classifier loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error(traceback.format_exc())

# Load models on startup
load_models()

# Emotion Labels
facial_emotions = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}
voice_emotions = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]

def fix_base64_padding(base64_string):
    """Fix Base64 padding errors by adding missing '=' characters."""
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    return base64_string

def process_facial_image(image_data):
    """Process base64 encoded facial image for emotion detection."""
    try:
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        image_data = fix_base64_padding(image_data)
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            raise ValueError("No face detected in image")

        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

        logger.info("✅ Face processed successfully")
        return roi_gray

    except Exception as e:
        logger.error(f"❌ Error processing facial image: {str(e)}")
        return None

def convert_webm_to_wav(webm_path, wav_path):
    """Convert a .webm audio file to .wav format."""
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")
    return wav_path

def process_voice_data(audio_bytes):
    """Process voice recording for emotion detection."""
    temp_webm = None
    temp_wav = None
    try:
        logger.info("🎵 Starting voice processing...")

        webm_path = os.path.join(TEMP_AUDIO_PATH, "recorded_audio.webm")
        with open(webm_path, "wb") as f:
            f.write(audio_bytes)

        logger.info(f"✅ WebM audio saved: {webm_path}")

        temp_wav = os.path.join(TEMP_AUDIO_PATH, "recorded_audio.wav")
        convert_webm_to_wav(webm_path, temp_wav)

        if not os.path.exists(temp_wav):
            raise FileNotFoundError(f"❌ WAV file not created: {temp_wav}")

        logger.info(f"✅ Successfully converted to WAV: {temp_wav}")

        y, sr = librosa.load(temp_wav, duration=5, sr=22050)
        logger.info(f"✅ Audio loaded: duration={len(y)/sr:.2f}s, sr={sr}Hz")

        target_length = 5 * sr
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )

        if mfcc.shape[1] > 94:
            mfcc = mfcc[:, :94]
        elif mfcc.shape[1] < 94:
            pad_width = 94 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        TRAINING_MEAN = -5
        TRAINING_STD = 10
        mfcc = (mfcc - TRAINING_MEAN) / (TRAINING_STD + 1e-8)

        features = np.expand_dims(mfcc, axis=0)
        logger.info(f"✅ Final features shape: {features.shape}")

        return features

    except Exception as e:
        logger.error(f"❌ Error processing voice data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        if temp_webm and os.path.exists(temp_webm):
            try:
                os.unlink(temp_webm)
                logger.info("✅ Temporary WebM file cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temporary WebM file: {str(e)}")
        
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
                logger.info("✅ Temporary WAV file cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temporary WAV file: {str(e)}")

# User Class and Authentication
class User(UserMixin):
    def __init__(self, id, name, email):
        self.id = str(id)
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    if not user_id:
        print("⚠️ load_user called with None user_id")
        return None

    print(f"🔍 load_user called with user_id: {user_id}")
    cursor = mysql.connection.cursor()

    try:
        cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()
        if user:
            print(f"✅ User Loaded: {user[0]}, {user[1]}, {user[2]}")
            return User(user[0], user[1], user[2])
    except Exception as e:
        print(f"❌ Error loading user: {e}")
    finally:
        cursor.close()

    print("⚠️ User Not Found!")
    return None

def verify_user(email, password):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[3], password):
            return User(user[0], user[1], user[2])
    except Exception as e:
        logger.error(f"❌ Error verifying user: {str(e)}")
    return None

def save_emotion_to_db(user_id, emotion_type, emotion, confidence):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute(
            'INSERT INTO emotion_history (user_id, emotion_type, emotion, confidence, timestamp) VALUES (%s, %s, %s, %s, NOW())',
            (user_id, emotion_type, emotion, confidence)
        )
        mysql.connection.commit()
        cursor.close()
        logger.info("✅ Emotion saved to database")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving emotion to database: {str(e)}")
        return False

# Routes
@app.route('/ensemble-prediction', methods=['POST'])
@login_required
def ensemble_prediction():
    """Combine facial and voice emotions with weights."""
    try:
        data = request.json
        facial_emotion = data.get('facial_emotion')
        voice_emotion = data.get('voice_emotion')
        facial_confidence = float(data.get('facial_confidence', 0))
        voice_confidence = float(data.get('voice_confidence', 0))
        
        if not all([facial_emotion, voice_emotion]):
            return jsonify({
                'status': 'error',
                'message': 'Both facial and voice emotions are required'
            }), 400
            
        final_emotion, confidence = calculate_ensemble_emotion(
            facial_emotion, 
            voice_emotion,
            facial_confidence,
            voice_confidence
        )
        
        # Save final emotion to database
        save_emotion_to_db(
            current_user.id,
            'ensemble',
            final_emotion,
            confidence
        )
        
        return jsonify({
            'status': 'success',
            'emotion': final_emotion,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"❌ Error in ensemble prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/detect-facial-emotion', methods=['POST'])
@login_required
def detect_facial_emotion():
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Invalid content type'}), 400

        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data received'}), 400

        face_roi = process_facial_image(image_data)
        if face_roi is None:
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 400

        prediction = facial_model.predict(face_roi)
        maxindex = int(np.argmax(prediction))
        emotion = facial_emotions[maxindex]
        confidence = float(prediction[0][maxindex])

        save_emotion_to_db(current_user.id, 'facial', emotion, confidence)

        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"❌ Error in facial emotion detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/detect-voice-emotion', methods=['POST'])
@login_required
def detect_voice_emotion():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file received'}), 400

        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'status': 'error', 'message': 'Invalid audio file'}), 400

        audio_bytes = audio_file.read()
        processed_features = process_voice_data(audio_bytes)

        if processed_features is None:
            return jsonify({'status': 'error', 'message': 'Failed to process audio'}), 400

        prediction = voice_model.predict(processed_features)
        maxindex = int(np.argmax(prediction))
        emotion = voice_emotions[maxindex]
        confidence = float(prediction[0][maxindex])

        save_emotion_to_db(current_user.id, 'voice', emotion, confidence)

        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"❌ Error in voice emotion detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([email, password]):
            flash('All fields are required!', 'error')
            return redirect(url_for('login'))

        user = verify_user(email, password)

        if user:
            login_user(user, remember=True)
            session["user_id"] = str(user.id)
            session.permanent = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')

        if not all([name, email, password, confirm_password]):
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        try:
            cursor = mysql.connection.cursor()
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            
            if cursor.fetchone():
                flash('Email already registered!', 'error')
                return redirect(url_for('register'))

            hashed_password = generate_password_hash(password)
            cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                           (name, email, hashed_password))
            mysql.connection.commit()
            cursor.close()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash('Registration failed! Please try again.', 'error')
            logger.error(f"Registration error: {e}")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('open_page'))

@app.route('/open')
def open_page():
    return render_template('open.html')

@app.route('/recommend/<emotion>', methods=['GET'])
def recommend_music(emotion):
    if emotion in playlists:
        return jsonify({"playlist": random.choice(playlists[emotion])})
    return jsonify({"error": "Emotion not found"}), 400

@app.route('/liked-playlists')
@login_required
def liked_playlists():
    """Render the Liked Playlists page with user-specific liked playlists from MySQL."""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT playlist_url FROM liked_playlists WHERE user_id = %s ORDER BY id DESC', (current_user.id,))
        user_playlists = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        logger.info(f"✅ Found {len(user_playlists)} playlists for user {current_user.id}")
        return render_template('liked_playlists.html', playlists=user_playlists)
    except Exception as e:
        logger.error(f"❌ Error fetching liked playlists: {str(e)}")
        flash('Error loading playlists', 'error')
        return render_template('liked_playlists.html', playlists=[])

@app.route('/like-playlist', methods=['POST'])
@login_required
def like_playlist():
    """API to add a playlist to the user's liked playlists."""
    playlist_url = request.json.get('playlist_url')
  # Ensure the key matches the JavaScript function

    if not playlist_url:
        return jsonify({"status": "error", "message": "Invalid playlist URL"}), 400

    try:
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT IGNORE INTO liked_playlists (user_id, playlist_url) VALUES (%s, %s)', 
                       (current_user.id, playlist_url))
        mysql.connection.commit()
        cursor.close()

        return jsonify({"status": "success", "message": "Playlist added successfully!"}), 200

    except Exception as e:
        logger.error(f"❌ Error liking playlist: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/clear-liked-playlists', methods=['POST'])
@login_required
def clear_liked_playlists():
    """Clear only the current user's liked playlists from MySQL."""
    try:
        logger.info(f"👤 User {current_user.id} attempting to clear playlists")
        
        cursor = mysql.connection.cursor()
        
        # Check if user has any playlists
        cursor.execute('SELECT COUNT(*) FROM liked_playlists WHERE user_id = %s', (current_user.id,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("⚠️ No playlists to clear")
            cursor.close()
            return jsonify({"message": "No playlists to clear!"}), 200
        
        # Delete playlists
        cursor.execute('DELETE FROM liked_playlists WHERE user_id = %s', (current_user.id,))
        mysql.connection.commit()
        cursor.close()
        
        logger.info(f"✅ Cleared {count} playlists")
        return jsonify({"message": f"Cleared {count} playlists successfully!"}), 200

    except Exception as e:
        logger.error(f"❌ Error clearing playlists: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-liked-playlists', methods=['GET'])
@login_required
def get_liked_playlists():
    """Return the current user's liked playlists from MySQL."""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT playlist_url FROM liked_playlists WHERE user_id = %s', (current_user.id,))
        user_playlists = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return jsonify({
            "status": "success",
            "count": len(user_playlists),
            "liked_playlists": user_playlists
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/debug-liked-playlists')
@login_required
def debug_liked_playlists():
    """Debug endpoint to check liked playlists table."""
    try:
        cursor = mysql.connection.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM liked_playlists')
        total_count = cursor.fetchone()[0]
        
        # Get user's playlists
        cursor.execute('SELECT * FROM liked_playlists WHERE user_id = %s', (current_user.id,))
        user_playlists = cursor.fetchall()
        
        # Get table structure
        cursor.execute('DESCRIBE liked_playlists')
        table_structure = cursor.fetchall()
        
        cursor.close()
        
        return jsonify({
            "total_playlists": total_count,
            "user_playlists_count": len(user_playlists),
            "user_id": current_user.id,
            "table_structure": [{"field": row[0], "type": row[1]} for row in table_structure],
            "sample_data": [{"id": row[0], "user_id": row[1], "url": row[2]} for row in user_playlists[:5]]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-db')
def test_db():
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()[0]
        cursor.close()
        return jsonify({"status": "success", "database": db_name})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

if __name__ == '__main__':
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
        )
    except Exception as e:
        logger.error(f"❌ Error starting server: {str(e)}")