# 🧠 Twitter Sentiment Analysis – AI-Powered Emotion Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 📝 Project Description
**Twitter Sentiment Analysis** is a comprehensive, multi-modal AI dashboard designed to detect and analyze emotional states across different media types. By leveraging state-of-the-art Natural Language Processing (NLP), Speech Recognition, and Computer Vision, the system provides deep insights into human emotions from **Text**, **Audio**, and **Video** inputs.

Whether it's analyzing a single tweet, a batch of customer feedback in a CSV, a voice recording, or a video clip, this dashboard offers real-time emotional intelligence with a high-performance, enterprise-grade user interface.

---

## 🚀 Features
- **🎯 Text Sentiment Analysis**: Predicts one of 10 nuanced emotional states (Happy, Sad, Angry, Fear, Surprise, etc.) with high confidence.
- **📂 Batch CSV Sentiment Analysis**: Process thousands of records instantly via CSV/TXT upload with progress tracking and downloadable results.
- **🎙️ Audio Emotion Detection**: Integrated with **Faster-Whisper AI** for lightning-fast speech-to-text transcription and subsequent emotional analysis.
- **🎥 Video Emotion Detection**: Utilizes **OpenCV**, **MediaPipe**, and **DeepFace** to detect facial expressions and aggregate emotional trends from video frames.
- **📊 Real-time Dashboard Analytics**: Interactive Plotly visualizations including emotion distributions, radar fingerprints, and confidence histograms.
- **🔐 Admin Authentication System**: Secure, role-based access control built with SQLite and Bcrypt password hashing.
- **💎 Enterprise Streamlit UI**: A modern, dark-themed "Glassmorphism" interface designed for a premium user experience.

---

## 🛠️ Tech Stack

### Frontend & UI
- **Streamlit**: Main application framework.
- **Custom CSS/HTML**: Glassmorphism design and custom animations.

### Backend & Database
- **Python 3.9+**: Core logic and orchestration.
- **SQLite**: User management and session data.
- **Bcrypt**: Secure credential hashing.

### Machine Learning (Text)
- **Scikit-learn**: Model training and evaluation.
- **LinearSVC**: Optimized linear support vector classification.
- **TF-IDF Vectorization**: Advanced text feature extraction with n-grams.

### Audio AI
- **Faster-Whisper**: High-performance speech recognition.
- **Librosa**: Audio preprocessing, normalization, and silence trimming.

### Computer Vision (Video)
- **MediaPipe Tasks API**: Robust face detection (Task 0.10.x compatible).
- **DeepFace**: Deep learning-based facial attribute analysis and emotion recognition.
- **OpenCV**: Frame extraction and image processing.

### Data & Visualization
- **Plotly**: Interactive charts and professional analytics.
- **Pandas & NumPy**: Efficient data manipulation and matrix operations.

---

## 🏗️ AI Architecture

The system operates through three primary pipelines:

1. **Text Pipeline**: 
   `Input Text` → `NLP Preprocessing (Cleaning/Lemmatization)` → `TF-IDF Vectorizer` → `LinearSVC Model` → `10-Class Emotion Prediction`

2. **Audio Pipeline**: 
   `Audio File` → `Librosa Preprocessing (Normalization/Resampling)` → `Whisper Speech Recognition` → `Extracted Text` → `Text Sentiment Model`

3. **Video Pipeline**: 
   `Video Stream` → `Frame Sampling` → `MediaPipe Face Detection` → `DeepFace Emotion Model` → `Weighted Emotion Aggregation`

---

## 📁 Project Structure

```text
Twitter-Sentiment-analysis/
│
├── app.py                # Main Streamlit Orchestrator
├── frontend/             # UI Components & Page Designs
│   ├── audio_page.py
│   ├── video_page.py
│   └── admin_login.py
├── backend/              # Core Logic & Processing Engines
│   ├── audio_processor.py
│   ├── video_processor.py
│   └── database.py
├── models/               # ML Models & Inference Engines
│   └── inference_engine.py
├── utils/                # Shared Utilities & Configurations
├── training/             # Model Training Scripts & Notebooks
├── data/                 # Sample Datasets & CSVs
├── requirements.txt      # Project Dependencies
└── README.md             # Project Documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/adrajameet7805/Twitter-Sentiment-analysis.git
   cd Twitter-Sentiment-analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📖 How to Use

1. **Launch**: Start the Streamlit app using the command above.
2. **Login**: Access the dashboard via the **Admin Access** (or navigate to `?page=login` internally).
3. **Analyze**: Use the sidebar to choose between **Single**, **Batch**, **Audio**, or **Video** analysis.
4. **Input**: Upload your file or type your text.
5. **Insights**: View the detailed emotion breakdown, confidence scores, and professional visualizations.

---

## 📈 Model Performance
*Current Text Sentiment Model (LinearSVC + TF-IDF):*

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 82% – 88% |
| **Precision** | 0.84 |
| **Recall** | 0.85 |
| **F1 Score** | 0.84 |

---

## 🔮 Future Improvements
- [ ] **Transformer-based models**: Migration to BERT or DistilBERT for better semantic context.
- [ ] **Real-time Twitter API**: Direct integration to pull live tweets by hashtag or user.
- [ ] **GPU Acceleration**: CUDA support for faster video and transcription processing.
- [ ] **Dockerization**: Containerized deployment for scalable production environments.
- [ ] **Mobile API**: Developing a FastAPI backend to serve as an endpoint for mobile applications.

---

## 👤 Author
**Meet Coding**  
- GitHub: [@adrajameet7805](https://github.com/adrajameet7805)
- Project: [Twitter Sentiment Analysis](https://github.com/adrajameet7805/Twitter-Sentiment-analysis)

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
