# Waggle - Early Detection of Canine Stress and Pre-Aggression

## Project Overview

Waggle is an AI-driven system for early detection of canine stress and pre-aggression signals from short videos. Using advanced computer vision and machine learning, Waggle analyzes dog body language to classify emotional states as friendly, nervous, or pre-aggressive. The system provides visual explanations highlighting key body cues (tail, posture, head orientation) that influence predictions, making it accessible to dog owners, trainers, and veterinarians.

**Status:** This is a prototype application developed during an internship at Elysium Technologies.

## Project Purpose and Scope

### Motivation

Dogs communicate emotions predominantly through subtle body language (tail position, ear orientation, posture, facial expressions). Early signs of stress or nervousness often precede aggression but are frequently missed by humans. Advanced computer vision and machine learning techniques now enable automatic detection of these subtle behavioral patterns in real-time, offering opportunities to improve human-dog interactions and prevent unsafe situations.

### Problem Statement

Subtle canine stress and pre-aggression cues are difficult for humans to detect consistently. Early behavioral signals often go unnoticed, potentially leading to preventable incidents. A real-time, explainable AI system can bridge this gap by automatically analyzing dog behavior and providing actionable insights.

### Technical Objectives

- Extract body keypoints and pose landmarks using MediaPipe Pose estimation
- Analyze temporal behavior patterns using LSTM deep learning models
- Classify dog emotional states into three categories: friendly, nervous, pre-aggressive
- Build and optimize multi-class machine learning classifiers
- Implement explainable AI to highlight key body movements influencing predictions
- Develop an end-to-end machine learning pipeline for video-to-prediction workflow

### Non-Technical Objectives

- Educate dog owners, trainers, and veterinarians about canine stress signals
- Provide accessible, real-time insights for improved human-dog safety
- Demonstrate practical AI applications in animal behavior analysis
- Bridge advanced machine learning techniques with real-world usability

## How It Works

1. **Video Input:** Upload a short dog video (MP4 format) through the Streamlit web interface
2. **Frame Extraction:** The video is automatically broken into individual frames
3. **Pose Estimation:** MediaPipe Pose detects body landmarks (tail, ears, head, limbs, posture)
4. **Feature Sequence Generation:** Temporal motion trends and keypoint sequences are extracted
5. **Behavior Classification:** An LSTM model analyzes the temporal patterns to predict emotional state
6. **Output:** Results include the predicted mood, confidence scores, and visual highlights of key body cues

## Features

- Upload and analyze dog videos directly through an intuitive web interface
- Extract 33 pose keypoints using MediaPipe for detailed body analysis
- Predict dog emotional states using an LSTM-based temporal model
- Visualize prediction confidence and mood distribution across video frames
- Explainable AI highlighting key body movements influencing predictions
- Real-time processing and visual feedback with pose overlays

## Technical Architecture

### System Components

**Video Processing and Preprocessing**
- Video input handling and frame extraction
- Image normalization and resizing
- Lighting adjustment and noise reduction
- Background segmentation to isolate the dog

**Feature Extraction**
- Pose keypoint detection: tail, ears, head, limbs, posture
- Temporal feature computation: motion changes over time
- Sequential representation for LSTM input

**Behavioral Prediction**
- LSTM model for temporal dependency analysis
- Multi-class classification (3 emotional states)
- Confidence scoring and prediction uncertainty quantification

**Explainability Module**
- Keypoint visualization overlaid on video frames
- Highlighting critical body parts influencing predictions
- Temporal attention visualization showing which frames drove the decision

**User Interface**
- Video upload functionality
- Real-time processing feedback
- Visual outputs with pose overlays
- Behavioral reports with timestamps and explanations

## Technologies and Libraries

### Core Libraries

- **Python 3.7+** - Programming language
- **Streamlit** - Web application framework for rapid deployment
- **MediaPipe** - Pose estimation and keypoint detection
- **OpenCV** - Video processing and frame extraction
- **TensorFlow/Keras** - Deep learning framework for LSTM models
- **Scikit-learn** - Machine learning utilities and model evaluation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Development Tools

- Git for version control
- Jupyter Notebooks for prototyping and analysis

## Model Architecture

### Classifier Comparison

| Metric | Random Forest | Logistic Regression | KNN | SVM |
|--------|---------------|-------------------|-----|-----|
| Accuracy | High and consistent | Low due to non-linearity | Sensitive to pose noise | Fair, needs tuning |
| Robustness | Good with noisy frames | Poor | Poor | Moderate |
| Computational Efficiency | Efficient | Efficient | Moderate | Expensive |
| Explainability | Feature importance insights | Limited | Limited | Limited |

**Selected Model:** Random Forest classifier for robust, efficient multi-class behavior classification with interpretable feature importance.

### LSTM for Temporal Analysis

LSTM (Long Short-Term Memory) networks are employed to analyze temporal dependencies in pose sequences, capturing behavioral patterns over time for more accurate state prediction.

## Dataset

The training dataset has been collected independently and is stored in the `data/` folder. The dataset includes:
- Manually annotated dog videos across multiple breeds
- Frame-level pose keypoint annotations
- Behavioral state labels (friendly, nervous, pre-aggressive)
- Diverse environmental conditions and dog poses

## Demo

A sample demo video is available in the repository:
- **Waggle.mp4** - Example video demonstrating the system's behavior classification capabilities. Upload this video through the Streamlit interface to see the model in action.

## Installation and Usage

### Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sairathna123/Waggle.git
   cd Waggle
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

### Usage

1. Upload a dog video (MP4 format) through the interface
2. Click "Analyze" to process the video
3. View the predicted emotional state and confidence scores
4. Examine visual explanations highlighting key body cues

## File Structure

```
Waggle/
├── app.py                 # Main Streamlit application
├── dog_pose_extractor.py  # Pose extraction utilities
├── requirements.txt       # Python dependencies
├── runtime.txt           # Runtime specifications
├── packages.txt          # System package dependencies
├── model/                # Pre-trained models and encoders
│   ├── model.pkl        # Trained classifier
│   └── encoder.pkl      # Label encoder
├── data/                 # Training dataset (self-collected)
├── Sample videos/        # Example videos for testing
└── README.md            # This file
```

## Results and Findings

Waggle has demonstrated reliable performance in classifying canine emotional states from video data. The system successfully:
- Extracts consistent pose landmarks across varied dog breeds and postures
- Accurately predicts emotional states with interpretable explanations
- Provides actionable insights for behavioral monitoring
- Operates with real-time performance suitable for practical deployment

## Future Work

- **Multimodal Analysis:** Integration of sound and voice analysis to study barking patterns and acoustic cues alongside visual data
- **Expanded Behavior Categories:** Extension to include playful, submissive, fearful, and relaxed states
- **Smart Camera Deployment:** Integration with security camera systems with intelligent alerting for behavior-related risks
- **Enhanced Explainability:** Implementation of heatmaps and improved visual highlights for clearer user understanding
- **Dataset Expansion:** Collection of larger and more diverse datasets to improve accuracy and enable multi-dog simultaneous monitoring
- **Model Optimization:** Fine-tuning for edge device deployment with minimal latency

## Conclusion

Waggle demonstrates the practical application of AI in animal behavior analysis, combining computer vision and deep learning to improve human-dog interactions. By providing explainable, real-time insights into canine emotional states, the system offers value to dog owners, shelters, trainers, and veterinarians. With its user-friendly interface and robust technical foundation, Waggle shows strong potential for real-world deployment in behavioral monitoring and safety applications.




