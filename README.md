# Face Liveness Detection - Android Application

A real-time face liveness detection Android app using **MiniFASNet ONNX models**, **ML Kit** for face detection, and **Traditional Computer Vision** techniques. The app determines whether a face presented to the camera is a **REAL** person or a **FAKE** (spoofing attempt - photo, video, or screen).

[Download the application by clicking on the following link :] (https://drive.google.com/file/d/1JI7X7lZwBudhiNsFWcGjjn7SIuXRvP9A/view?usp=sharing)

## 🚀 Features

- **MiniFASNet ONNX Models**: Uses pre-trained Silent-Face-Anti-Spoofing MiniFASNet models
- **Multi-Model Fusion**: Combines CNN + traditional CV methods for robust detection
- **ML Kit Face Detection**: Fast and accurate face detection using Google's ML Kit
- **Real-time Processing**: Efficient ONNX Runtime inference on mobile
- **Modern UI**: Built with Jetpack Compose and Material3
- **Attack Type Detection**: Identifies specific attack types (photo, video, screen, mask)

## 🧠 Model Architecture

The app uses a **hybrid approach** combining deep learning (MiniFASNet) with traditional computer vision methods for robust liveness detection.

### MiniFASNet Models

| Model File | Scale | Architecture |
|------------|-------|--------------|
| `2.7_80x80_MiniFASNetV2.onnx` | 2.7x | MiniFASNet V2 |
| `4_0_0_80x80_MiniFASNetV1SE.onnx` | 4.0x | MiniFASNet V1 with SE blocks |

### Classification Labels

| Label Index | Class | Result |
|-------------|-------|--------|
| 0 | Paper Attack | FAKE |
| 1 | Real Face | REAL |
| 2 | Screen/Video Attack | FAKE |

---

## 🔬 Traditional CV Methods

The app combines multiple traditional computer vision techniques to enhance detection accuracy:

### 1. LBP (Local Binary Pattern) - Texture Analysis
- **Weight**: 12%
- **Purpose**: Detects texture uniformity differences between real faces and prints
- **How it works**: Calculates 8-neighbor LBP histogram and measures entropy
- **Detection**: Real faces have natural texture variation (entropy 0.6-0.9); printed photos appear too uniform

### 2. Frequency Analysis - Moiré Pattern Detection
- **Weight**: 15%
- **Purpose**: Detects moiré patterns common when photographing screens
- **How it works**: Calculates gradient-based high/low frequency energy ratio
- **Detection**: Screen replays show characteristic high-frequency periodic patterns

### 3. Color Analysis - Skin Tone Validation
- **Weight**: 8%
- **Purpose**: Validates natural skin color distribution
- **How it works**: Converts to YCrCb color space and checks skin pixel ratio
- **Detection**: Real faces have 40-75% skin pixels in typical Cr/Cb ranges; masks and unusual lighting fail this test

### 4. Motion Analysis - Temporal Pattern Detection
- **Weight**: 10%
- **Purpose**: Detects video replay attacks through motion patterns
- **How it works**: Frame differencing with variance analysis across 10-frame history
- **Detection**: Real faces show natural, variable motion; videos have smooth/periodic patterns; photos show no motion

### 5. Texture Sharpness - Laplacian Analysis
- **Weight**: 5%
- **Purpose**: Detects blur and artificial sharpening artifacts
- **How it works**: Calculates Laplacian variance (2nd derivative) for edge detection
- **Detection**: Real faces have natural sharpness; blurry photos and artificially enhanced images fail

### 6. Depth Analysis - Head Pose Variation
- **Weight**: 10%
- **Purpose**: Validates 3D presence through head movement
- **How it works**: Tracks Euler angle (yaw, pitch, roll) variance over 20 frames
- **Detection**: Real 3D faces show natural head movement variance; 2D attacks (photos/screens) are static

---

## 🎯 Fusion Scoring System

The final liveness score is calculated using a **weighted average** of all detection methods:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DETECTION METHODS                               │
├─────────────────────────────────────────────────────────────────────┤
│  MiniFASNet CNN (40%)  │  LBP (12%)  │  Frequency (15%)  │  ...    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      WEIGHTED FUSION                                 │
│                                                                      │
│  Final Score = 0.40 × CNN + 0.12 × LBP + 0.15 × Frequency +         │
│                0.08 × Color + 0.10 × Motion + 0.05 × Texture +      │
│                0.10 × Depth                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DECISION THRESHOLDS                             │
│                                                                      │
│  Score ≥ 0.60  →  PASS (Real Face)                                  │
│  Score ≤ 0.40  →  FAIL (Spoof Detected)                             │
│  0.40 < Score < 0.60  →  UNCERTAIN (Active Liveness Required)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Weights

| Method | Weight | Primary Detection Target |
|--------|--------|--------------------------|
| **MiniFASNet CNN** | 40% | All attack types |
| **Frequency Analysis** | 15% | Screen/video moiré patterns |
| **LBP Texture** | 12% | Printed photos |
| **Motion Analysis** | 10% | Video replays |
| **Depth Analysis** | 10% | 2D attacks (photos/screens) |
| **Color Analysis** | 8% | Masks, abnormal lighting |
| **Texture Sharpness** | 5% | Blur/enhancement artifacts |

### Attack Type Detection

Based on individual score patterns, the system identifies specific attack types:

| Attack Type | Detection Criteria |
|-------------|-------------------|
| **Screen Replay** | Low CNN score + Low frequency score (moiré) |
| **Printed Photo** | Low LBP score + Low motion score |
| **Video Replay** | Low motion variance |
| **2D Mask** | Low depth score |
| **3D Mask** | Abnormal color + texture scores |

---

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
│   │   ├── TraditionalCVAnalyzer.kt     # LBP, Frequency, Color, Texture
│   │   ├── MotionAnalyzer.kt            # Motion + Depth analysis
│   │   ├── FusionScorer.kt              # Weighted score fusion
│   │   ├── BlinkDetector.kt             # Active liveness (blink)
│   │   └── CNNInference.kt
│   └── ui/
│       ├── MainActivity.kt
│       ├── LivenessViewModel.kt
│       ├── CameraPreview.kt
│       ├── LivenessDetectionScreen.kt
│       └── theme/Theme.kt
└── res/
    └── ...
```

---

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

---

## 📖 Usage

1. Launch the app
2. Grant camera permission when prompted
3. Position your face within the camera frame
4. The app will automatically:
   - Detect your face using ML Kit
   - Run MiniFASNet CNN inference
   - Calculate traditional CV scores
   - Compute weighted fusion score
5. View the result: **REAL** (live person) or **FAKE** (spoofing attempt)
6. If uncertain, follow active liveness prompt (blink detection)

---

## 📚 References

- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) - Original MiniFASNet implementation
- [ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Local Binary Patterns (LBP)](https://en.wikipedia.org/wiki/Local_binary_patterns)

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- MiniVision AI for the Silent-Face-Anti-Spoofing and MiniFASNet models
- Google ML Kit team for face detection
- Microsoft for ONNX Runtime
