# Face Liveness Detection - Android Application

A hybrid two-stage face liveness detection system for Android that detects whether a face presented to the camera is a live person or a spoofing attempt.

## Features

- **Passive Detection**: Silent analysis using CNN + Traditional CV methods
- **Active Detection**: Blink-based verification for uncertain cases
- **Multi-Attack Detection**: Detects photos, videos, screens, masks, and deepfakes
- **Real-time Processing**: Fast inference with TensorFlow Lite
- **Modern UI**: Built with Jetpack Compose

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 PASSIVE DETECTION                    │
├─────────────────────────────────────────────────────┤
│  CNN (35%)  │  LBP (15%)  │  Motion (15%)  │  ...  │
└─────────────────────────────────────────────────────┘
                         ↓
              [Score >= 0.85] → PASS
              [Score <= 0.55] → FAIL
              [Uncertain] → Active Detection
                         ↓
┌─────────────────────────────────────────────────────┐
│                 ACTIVE DETECTION                     │
│              Blink Detection (2 blinks)             │
└─────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 26+ (Android 8.0)
- Kotlin 1.9.20

### Installation

1. Clone/Copy the project to your workspace
2. Open in Android Studio
3. **Important**: Add the TFLite model to `app/src/main/assets/`:
   - Download Silent-Face-Anti-Spoofing model
   - Convert to TFLite format
   - Name it `silent_face_model.tflite`
4. Sync Gradle and build

### Dependencies

- **Jetpack Compose**: Modern UI toolkit
- **CameraX**: Camera handling
- **ML Kit**: Face detection and face mesh
- **TensorFlow Lite**: CNN inference
- **OpenCV**: Traditional CV algorithms

## Project Structure

```
app/src/main/java/com/uidai/livenessdetection/
├── domain/
│   ├── models/
│   │   ├── LivenessResult.kt    # Data models
│   │   └── FaceData.kt
│   └── LivenessDetector.kt      # Main orchestrator
├── ml/
│   ├── CNNInference.kt          # TFLite inference
│   ├── TraditionalCVAnalyzer.kt # LBP, Fourier, Color, Texture
│   ├── MotionAnalyzer.kt        # Optical flow
│   ├── BlinkDetector.kt         # EAR-based blink detection
│   └── FusionScorer.kt          # Score aggregation
└── ui/
    ├── MainActivity.kt
    ├── LivenessViewModel.kt
    ├── CameraPreview.kt
    ├── LivenessDetectionScreen.kt
    └── theme/Theme.kt
```

## Detection Methods

| Method | Weight | Detects |
|--------|--------|---------|
| CNN (Silent-Face) | 35% | All attack types |
| LBP Texture | 15% | Printed photos |
| Motion Analysis | 15% | Video replays |
| Frequency Analysis | 10% | Screen moiré |
| Color Analysis | 10% | Mask materials |
| Texture Analysis | 10% | Print/screen artifacts |
| Depth Analysis | 5% | 2D attacks |

## Usage

1. Launch the app
2. Grant camera permission when prompted
3. Position your face in the camera frame
4. Hold steady during passive analysis (2-3 seconds)
5. If prompted, blink twice naturally
6. View the verification result


## License

This project is for educational purposes.

## Acknowledgments

- Silent-Face-Anti-Spoofing for the CNN architecture reference
- Google ML Kit for face detection
- OpenCV for traditional CV algorithms
