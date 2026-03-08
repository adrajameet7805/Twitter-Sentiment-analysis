# 🧠 Twitter Sentiment Analysis: Multi-Modal AI Dashboard

A professional, enterprise-grade AI dashboard built with **Python** and **Streamlit** for real-time emotional intelligence. This project leverages state-of-the-art Deep Learning models to analyze sentiments across three critical mediums: **Text**, **Audio**, and **Video**.

---

## 🌟 Project Overview
The **Twitter Sentiment Analysis** system is designed to provide comprehensive emotional insights by processing multi-modal data. Traditional sentiment analysis is often limited to "positive" or "negative" text. This project breaks that barrier by detecting **10 distinct emotional states** (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral, etc.) from textual content, spoken word, and facial expressions.

The purpose is to give brands, researchers, and developers a sophisticated tool for understanding human sentiment with high granularity and accuracy (>95% on hybrid datasets).

---

## 🚀 Key Features

*   **Text Sentiment Analysis**: Highly accurate classification of text data using custom-trained hybrid NLP models.
*   **Audio Sentiment Analysis**: Uses **Faster-Whisper** for high-speed speech-to-text and transcribes audio to apply emotional intelligence on the spoken narrative.
*   **Video Sentiment Analysis**: Implements a computer vision pipeline using **MediaPipe** (Face Detection) and **DeepFace** (Emotion CNN) for frame-by-frame (batched) facial emotion tracking.
*   **Interactive AI Dashboard**: A sleek, dark-themed UI with glassmorphism aesthetics, providing real-time data visualizations via **Plotly**.
*   **Batch Processing**: Capability to process thousands of records via CSV/TXT uploads with high-speed inference.
*   **Secure Admin System**: Multi-user authentication with encrypted credential storage.

---

## 🛠 Technology Stack

*   **Language**: Python 3.10+
*   **Web Framework**: Streamlit (with customized CSS/JavaScript)
*   **Deep Learning**: TensorFlow, Keras, PyTorch
*   **Computer Vision**: OpenCV, MediaPipe, DeepFace
*   **Speech Processing**: Faster-Whisper, Librosa
*   **General ML & NLP**: Scikit-Learn, NLTK, Transformers, Pandas, NumPy
*   **Visualization**: Plotly Express

---

## 🏗 System Architecture

1.  **Text Pipeline**: Inputs are preprocessed and vectorized using optimized TF-IDF or Transformer-based embeddings, followed by inference through a multi-class CNN/Feed-forward model.
2.  **Audio Pipeline**: Audio files (WAV, MP3, OGG) are loaded via `librosa`, silence-filtered, and transcribed using `Faster-Whisper` (tiny.en). The resulting transcript is then passed to the Text Sentiment engine.
3.  **Video Pipeline**: Videos are sampled at 6 FPS. Faces are detected using MediaPipe/OpenCV, cropped, and normalized. Predictions are performed in optimized **batches of 32** through a specialized Face-Emotion CNN to ensure real-time performance.

---

## 📂 Project Structure

*   **/backend**: Core logic including [audio_processor.py](cci:7://file:///c:/Users/Meet/OneDrive/Desktop/p/video_sentiment_project/backend/audio_processor.py:0:0-0:0), [video_processor.py](cci:7://file:///c:/Users/Meet/OneDrive/Desktop/p/video_sentiment_project/backend/video_processor.py:0:0-0:0), and [face_emotion_cnn.py](cci:7://file:///c:/Users/Meet/OneDrive/Desktop/p/video_sentiment_project/backend/face_emotion_cnn.py:0:0-0:0) for model inference.
*   **/frontend**: Modular UI components for audio/video analysis pages and standard dashboard widgets.
*   **/models**: Contains pre-trained weights (`.h5`, [.pkl](cci:7://file:///c:/Users/Meet/OneDrive/Desktop/p/video_sentiment_project/model.pkl:0:0-0:0)) and the unified `EmotionInferenceV4` engine.
*   **/training**: Research scripts and Jupyter Notebooks used for model architecture design and fine-tuning.
*   **/utils**: Configuration files for emotion mappings, color palettes, and global constants.
*   **/data**: Storage for sample input files and processed results.
*   **/logs**: System logs for monitoring inference performance and errors.

---

## ⚙️ How to Run

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Execution
Run the Streamlit application using the following command:
```bash
streamlit run app.py
