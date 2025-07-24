# 🎵 MoodTunes – Emotion-Based Music Recommendation System

**MoodTunes** is a real-time music recommendation system that detects a user's mood using both **facial expressions** and **voice tone**, then suggests appropriate YouTube Music playlists.

It uses a **hybrid ensemble model** combining **CNN (Convolutional Neural Network)** for facial emotion recognition and **LSTM (Long Short-Term Memory)** for voice emotion recognition. The final emotion decision is based on combining both predictions to improve accuracy and personalization.

---

## 📌 Features

- 🎥 Real-time facial emotion detection via webcam
- 🎤 Voice emotion detection from recorded speech
- 🧠 **Ensemble-based emotion classification** using CNN and LSTM
- 🎶 Music recommendations via curated YouTube playlists
- ❤️ Users can like playlists and revisit them later

---

## 🤖 Ensemble Learning Approach

MoodTunes uses a **dual-model ensemble strategy**:
- **CNN Model** trained on facial images from the **FER-2013 dataset**
- **LSTM Model** trained on audio features (MFCCs) from **TESS** and **RAVDESS** datasets

Each model independently predicts emotion:
- The system then selects the **dominant emotion** using a rule-based logic or a confidence-weighted scheme (e.g., match if both agree, otherwise fallback to priority or highest confidence).

This **ensemble improves accuracy** and captures emotional cues more effectively than using a single input type.

---

## 🔧 Tech Stack

- **Python**, **Flask**
- **OpenCV** for image capture
- **TensorFlow/Keras** for deep learning
- **Librosa** for audio processing
- **YouTube Music** for curated playlist recommendations

---

## 📊 Emotion Classes

- Happy
- Sad
- Angry
- Neutral
- Surprised
- Fearful

---

## 🗂️ Datasets Used

### Facial Emotion Detection
- **FER-2013**
  - Grayscale 48x48 images
  - Preprocessed, normalized, one-hot encoded
  - Model: CNN

### Voice Emotion Detection
- **TESS** and **RAVDESS**
  - WAV audio files, preprocessed (resampled, normalized)
  - MFCC feature extraction
  - Model: LSTM

---

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| CNN (Facial) | 87% |
| LSTM (Voice) | 83% |

**Combined Ensemble** leads to improved reliability across real-world emotional variation.

---

## 🚀 How It Works

1. **User’s face** is captured via webcam
2. **Voice sample** is recorded via mic
3. CNN and LSTM models classify emotions independently
4. Predictions are **combined in an ensemble strategy**
5. A **YouTube Music playlist** matching the detected emotion is displayed
6. Users can **like playlists** to save them for later

---

## 🧪 Future Scope

- Integrate Spotify/Apple Music APIs
- AI-powered personalization based on emotional history
- Deploy as mobile app
- Use soft-voting or train a meta-classifier for ensemble fusion

---

## 👩‍💻 Author

**Alina M V**  

---

## 📜 License

This project is for academic and demonstration purposes only.
