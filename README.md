# Face Liveness Detection - Android Application

A real-time face liveness detection Android app using **MiniFASNet ONNX models** and **ML Kit** for face detection. The app determines whether a face presented to the camera is a **REAL** person or a **FAKE** (spoofing attempt - photo, video, or screen).

## 🚀 Features

- **MiniFASNet ONNX Models**: Uses pre-trained Silent-Face-Anti-Spoofing MiniFASNet models
- **Multi-Model Fusion**: Combines predictions from multiple scales (2.7x and 4.0x) for robust detection
- **ML Kit Face Detection**: Fast and accurate face detection using Google's ML Kit
- **Real-time Processing**: Efficient ONNX Runtime inference on mobile
- **Modern UI**: Built with Jetpack Compose and Material3
- **Binary Output**: Clean REAL/FAKE classification with confidence scores

## 🧠 Model Architecture

The app uses **MiniFASNet** (Mini Face Anti-Spoofing Network) - a lightweight CNN designed for mobile deployment.

### Models Used

| Model File | Scale | Architecture |
|------------|-------|--------------|
| `2.7_80x80_MiniFASNetV2.onnx` | 2.7x | MiniFASNet V2 |
| `4_0_0_80x80_MiniFASNetV1SE.onnx` | 4.0x | MiniFASNet V1 with SE blocks |

### Detection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Kit Face Detection                    │
│              (Detects face bounding box)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Face Box Expansion                          │
│       (Scale 2.7x and 4.0x for different contexts)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              MiniFASNet ONNX Inference                       │
│     Input: 80x80 BGR image → Output: [paper, real, screen]  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Multi-Model Score Fusion                      │
│         Average softmax scores across all models            │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    REAL or FAKE + Confidence
```

### Classification Labels

| Label Index | Class | Result |
|-------------|-------|--------|
| 0 | Paper Attack | FAKE |
| 1 | Real Face | REAL |
| 2 | Screen/Video Attack | FAKE |

## 📁 Project Structure

```
app/src/main/
├── assets/
│   ├── 2.7_80x80_MiniFASNetV2.onnx      # MiniFASNet V2 model
│   ├── 4_0_0_80x80_MiniFASNetV1SE.onnx  # MiniFASNet V1-SE model
│   └── silent_face_model.tflite         # Alternative TFLite model
├── java/com/uidai/livenessdetection/
│   ├── domain/
│   │   ├── models/
│   │   │   ├── LivenessResult.kt
│   │   │   └── FaceData.kt
│   │   └── LivenessDetector.kt
│   ├── ml/
│   │   ├── OnnxLivenessDetector.kt      # ONNX inference engine
│   │   ├── CNNInference.kt
│   │   ├── BlinkDetector.kt
│   │   ├── MotionAnalyzer.kt
│   │   ├── TraditionalCVAnalyzer.kt
│   │   └── FusionScorer.kt
│   └── ui/
│       ├── MainActivity.kt
│       ├── LivenessViewModel.kt
│       ├── CameraPreview.kt
│       ├── LivenessDetectionScreen.kt
│       └── theme/Theme.kt
└── res/
    └── ...
```

## 🛠️ Setup

### Prerequisites
- Android Studio Hedgehog (2023.1.1) or later
- Android SDK 24+ (Android 7.0)
- Kotlin 1.9.20+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/viswarup/sitaa-face-liveliness.git
   ```

2. Open the project in Android Studio

3. Sync Gradle and wait for dependencies to download

4. Connect an Android device or start an emulator

5. Run the app

### Dependencies

- **ONNX Runtime**: `com.microsoft.onnxruntime:onnxruntime-android:1.16.3`
- **ML Kit Face Detection**: `com.google.mlkit:face-detection:16.1.6`
- **CameraX**: For camera feed handling
- **Jetpack Compose**: Modern declarative UI

## 📖 Usage

1. Launch the app
2. Grant camera permission when prompted
3. Position your face within the camera frame
4. The app will automatically detect your face and run liveness check
5. View the result: **REAL** (live person) or **FAKE** (spoofing attempt)

## 🔬 Technical Details

### Preprocessing Steps
1. **Face Detection**: ML Kit detects face bounding box
2. **Box Expansion**: Expand bounding box by scale factor (2.7x or 4.0x)
3. **Crop & Resize**: Extract face region and resize to 80x80 pixels
4. **Channel Order**: Convert to BGR format (matching Python training)
5. **CHW Format**: Reshape to [1, 3, 80, 80] tensor

### Inference
- ONNX Runtime executes MiniFASNet models
- Output: 3-class softmax scores [paper, real, screen]
- Multi-model fusion averages scores across all models

## 📚 References

- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) - Original MiniFASNet implementation
- [ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection)
- [ONNX Runtime](https://onnxruntime.ai/)

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- MiniVision AI for the Silent-Face-Anti-Spoofing and MiniFASNet models
- Google ML Kit team for face detection
- Microsoft for ONNX Runtime
