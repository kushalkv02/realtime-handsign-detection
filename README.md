# realtime-handsign-detection

# Sign Language Detector

A real-time hand gesture recognition system for sign language using computer vision and deep learning. This project enables users to collect hand gesture data, train classification models, and perform real-time sign language detection with audio feedback.

## Features
- **Data Collection:** Capture hand landmark data for custom gestures using a webcam and Mediapipe.
- **Model Training:** Train and evaluate multiple classifiers (Logistic Regression, Random Forest, Neural Network) on collected gesture data.
- **Real-Time Prediction:** Recognize gestures live from webcam feed and speak out detected signs using text-to-speech.

## Project Structure
```
gesture_classifier.h5         # Trained neural network model
label_encoder.pkl             # Label encoder for gesture classes
main.ipynb                    # Notebook for data collection and visualization
train_models.ipynb            # Notebook for model training and evaluation
realtime_predict.py            # Real-time gesture recognition script
data/
    ThumbsUp.csv              # Example gesture data
    V.csv                     # Example gesture data
```

## Setup Instructions
1. **Install Dependencies**
   - Python 3.7+
   - Required packages: `opencv-python`, `mediapipe`, `numpy`, `tensorflow`, `scikit-learn`, `pyttsx3`, `joblib`, `matplotlib`, `pandas`
   - Install with pip:
     ```bash
     pip install opencv-python mediapipe numpy tensorflow scikit-learn pyttsx3 joblib matplotlib pandas
     ```

2. **Collect Gesture Data**
   - Open `main.ipynb` and run the data collection cell.
   - Change the `gesture_label` variable for each sign and press 's' to save samples.
   - Data is saved in `data/<gesture_label>.csv`.

3. **Train Models**
   - Open `train_models.ipynb`.
   - Run all cells to clean data, train models, and save the best model (`gesture_classifier.h5`) and label encoder (`label_encoder.pkl`).

4. **Real-Time Prediction**
   - Run `realtime_predict.py`:
     ```bash
     python realtime_predict.py
     ```
   - The script will start webcam-based gesture recognition and speak out detected signs.

## How It Works
- **Hand Detection:** Uses Mediapipe to extract 21 hand landmarks (x, y, z) per frame.
- **Classification:** Trained neural network predicts gesture class from landmark data.
- **Smoothing:** Uses a rolling window to smooth predictions for robust output.
- **Audio Feedback:** Detected gesture is spoken aloud using text-to-speech.

## Customization
- Add new gestures by collecting more data and retraining the model.
- Adjust model architecture or parameters in `train_models.ipynb` for improved accuracy.

## References
- [Mediapipe](https://google.github.io/mediapipe/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)

## License
MIT License
