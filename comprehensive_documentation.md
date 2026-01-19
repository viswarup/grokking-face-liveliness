# Face Liveness Detection System - Complete Technical Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Why This Approach?](#why-this-approach)
4. [Models & Algorithms](#models-and-algorithms)
5. [Complete Code Flow](#complete-code-flow)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation)
7. [Decision Logic](#decision-logic)
8. [Performance Characteristics](#performance-characteristics)
9. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### What Are We Building?
An Android mobile application that detects whether a face presented to the camera is a **live person** or a **spoofing attempt** (photo, video, mask, deepfake).

### Core Requirements (from UIDAI)
- ✅ Detect physical attacks (photos, videos, masks)
- ✅ Detect digital attacks (deepfakes, adversarial)
- ✅ Work on both enrolment (PC) and authentication (mobile)
- ✅ Minimize user friction (passive-first approach)
- ✅ Model size ≤ 6MB
- ✅ Latency ≤ 1-1.5 seconds
- ✅ APCER ≤ 0.10%, BPCER ≤ 0.10%

### Our Solution
A **hybrid two-stage system**:
1. **Stage 1 (Passive):** Analyze video silently using CNN + Traditional CV
2. **Stage 2 (Active):** If uncertain, ask user to blink twice

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ANDROID APPLICATION                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         USER INTERFACE (Jetpack Compose)       │    │
│  │  - Camera Preview (60% screen)                 │    │
│  │  - Status Panel (40% screen)                   │    │
│  │  - Real-time bounding box overlay              │    │
│  └────────────────────────────────────────────────┘    │
│                         ↕                               │
│  ┌────────────────────────────────────────────────┐    │
│  │           VIEWMODEL (State Management)         │    │
│  │  - Manages UI state                            │    │
│  │  - Coordinates frame processing                │    │
│  └────────────────────────────────────────────────┘    │
│                         ↕                               │
│  ┌────────────────────────────────────────────────┐    │
│  │        LIVENESS DETECTOR (Orchestrator)        │    │
│  │  - Coordinates all detection modules           │    │
│  │  - Implements decision logic                   │    │
│  │  - Aggregates scores                           │    │
│  └────────────────────────────────────────────────┘    │
│                         ↕                               │
│  ┌─────────────────────────────────────────────────────┤
│  │          DETECTION MODULES (Parallel)          │    │
│  ├─────────────────────────────────────────────────────┤
│  │                                                 │    │
│  │  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │ CNN Module   │  │ Traditional CV       │   │    │
│  │  │              │  │ - LBP Texture        │   │    │
│  │  │ Silent-Face  │  │ - Fourier Analysis   │   │    │
│  │  │ (TFLite)     │  │ - Color Analysis     │   │    │
│  │  │              │  │ - Edge Sharpness     │   │    │
│  │  └──────────────┘  └──────────────────────┘   │    │
│  │                                                 │    │
│  │  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │ Motion       │  │ Depth Analysis       │   │    │
│  │  │ Analysis     │  │                      │   │    │
│  │  │              │  │ Euler Angle          │   │    │
│  │  │ Optical Flow │  │ Tracking             │   │    │
│  │  └──────────────┘  └──────────────────────┘   │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────────┐ │    │
│  │  │ Active Liveness (Blink Detection)       │ │    │
│  │  │ Eye Aspect Ratio (EAR) Method            │ │    │
│  │  └──────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┤
│                         ↕                               │
│  ┌────────────────────────────────────────────────┐    │
│  │           FUSION SCORER                        │    │
│  │  - Weighted combination of all scores          │    │
│  │  - Decision thresholds (PASS/FAIL/UNCERTAIN)   │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
├─────────────────────────────────────────────────────────┤
│              DEVICE CAPABILITIES                        │
│  - CameraX (Front Camera)                               │
│  - ML Kit (Face Detection + Face Mesh)                  │
│  - TensorFlow Lite (CNN Inference)                      │
│  - OpenCV (Computer Vision)                             │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Camera Frame
    ↓
Face Detection (ML Kit)
    ↓
Face Found? → NO → Show "Align face" message
    ↓ YES
Extract:
- Face Bitmap (224x224)
- 468 Facial Landmarks
- Euler Angles (yaw, pitch, roll)
    ↓
┌───────────────── PARALLEL PROCESSING ─────────────────┐
│                                                        │
│  CNN           LBP          Fourier      Color        │
│  Analysis      Texture      Analysis     Analysis     │
│  ↓             ↓            ↓            ↓            │
│  score_cnn     score_lbp    score_freq   score_color │
│                                                        │
│  Motion        Edge         Depth                     │
│  Analysis      Sharpness    Analysis                  │
│  ↓             ↓            ↓                         │
│  score_motion  score_text   score_depth               │
│                                                        │
└────────────────────────────────────────────────────────┘
    ↓
Fusion Scorer
    ↓
final_score = weighted_sum(all_scores)
    ↓
Decision Logic:
├─ final_score ≥ 0.85 → PASS (Live Face)
├─ final_score ≤ 0.55 → FAIL (Spoof Detected)
└─ 0.55 < score < 0.85 → UNCERTAIN (Trigger Blink)
    ↓
If UNCERTAIN:
    ↓
Active Liveness (Blink Detection)
    ↓
Detect 2 natural blinks → PASS/FAIL
```

---

## Why This Approach?

### Design Philosophy: Defense in Depth

**Problem:** No single detection method is perfect.
- CNNs can be fooled by high-quality deepfakes
- Traditional CV fails on sophisticated attacks
- Motion analysis alone misses static attacks

**Solution:** Use multiple complementary methods, each catching what others miss.

### Why Hybrid (CNN + Traditional CV)?

| Method | Strengths | Weaknesses | Use Case |
|--------|-----------|------------|----------|
| **CNN (Deep Learning)** | Learns complex patterns, good on deepfakes | Can be fooled, needs data | Primary detection |
| **LBP (Local Binary Patterns)** | Fast, catches printed photos | Poor on videos | Texture validation |
| **Fourier Analysis** | Detects screen moiré patterns | Limited scope | Screen replay detection |
| **Color Analysis** | Simple, effective | Lighting dependent | Material detection |
| **Optical Flow** | Catches unnatural motion | Noisy | Video replay detection |
| **Depth Tracking** | Detects 2D attacks | Needs movement | 3D validation |
| **Blink Detection** | Foolproof for masks | User friction | Final fallback |

**Combined:** These methods cover each other's blind spots.

### Why Two-Stage (Passive → Active)?

**User Experience Priority:**
- 85% of cases: Passive detection is confident → No user action needed
- 10% of cases: Sophisticated attacks → Active prompt needed
- 5% of cases: Genuine users in poor conditions → Active prompt helps

**Benefit:** Minimize friction for most users while maintaining security.

---

## Models and Algorithms

### 1. CNN Model: Silent-Face-Anti-Spoofing

**What is it?**
A deep learning model specifically trained to detect face spoofing attacks.

**Architecture:**
- Base: MobileNetV2 (lightweight, mobile-optimized)
- Input: 224×224 RGB image
- Output: 2 classes [spoof_probability, real_probability]
- Size: ~2.1MB (quantized TFLite)

**Why Silent-Face?**
- ✅ Pre-trained on multiple datasets (CASIA-FASD, OULU-NPU, Replay-Attack)
- ✅ Already optimized for mobile
- ✅ Open source, proven in competitions
- ✅ Handles most common attacks (photos, basic videos, masks)

**Training Data (what it has seen):**
- Printed photos (various paper types)
- Video replays (phone/tablet screens)
- 2D/3D masks
- Different lighting conditions
- Multiple ethnicities

**What it detects:**
- Print texture patterns
- Screen artifacts
- Mask materials
- Unnatural facial features

**Code Location:** `CNNInference.kt`

**How it works:**
```kotlin
// 1. Preprocess image
resize(face_bitmap, 224x224)
normalize(pixel_values, [-1, 1])

// 2. Run inference
output = tflite_model.predict(preprocessed_image)
// output = [0.15, 0.85] means 85% confidence real face

// 3. Return score
return output[1]  // real_probability
```

---

### 2. LBP (Local Binary Patterns)

**What is it?**
A texture descriptor that encodes local patterns around each pixel.

**Why use it?**
Printed photos have uniform textures that LBPs can detect instantly.

**How it works:**
```
For each pixel:
1. Compare with 8 neighbors
2. Create binary code (1 if neighbor ≥ center, 0 otherwise)
3. Convert to number (0-255)

Example:
Neighbors: [120, 130, 110, 125, 140, 135, 115, 128]
Center: 125

Binary: [0, 1, 0, 1, 1, 1, 0, 1] → 93

4. Build histogram of these codes
5. Real faces: diverse patterns
   Printed photos: repetitive patterns
```

**What it detects:**
- ✅ Printed photos (too uniform)
- ✅ Paper texture
- ✅ Low-quality masks
- ❌ High-quality screens (similar to real)

**Speed:** 3-5ms per frame

**Code Location:** `TraditionalCVAnalyzer.kt` → `calculateLBPScore()`

---

### 3. Fourier Frequency Analysis

**What is it?**
Converts image from spatial domain to frequency domain to detect periodic patterns.

**Why use it?**
Screens have refresh rates (50-60Hz) that create moiré patterns invisible to human eyes.

**How it works:**
```
1. Convert image to grayscale
2. Apply 2D Fast Fourier Transform (FFT)
3. Get magnitude spectrum
4. Look for periodic peaks in 50-60Hz range
5. If found → Screen replay detected
```

**Moiré Pattern Explanation:**
```
Camera sensor grid + Screen pixel grid = Interference pattern
(Similar to when you photograph a TV screen and see waves)
```

**What it detects:**
- ✅ Phone screen replays
- ✅ Tablet replays
- ✅ Monitor replays
- ❌ Printed photos (no refresh rate)

**Speed:** 5-8ms per frame

**Code Location:** `TraditionalCVAnalyzer.kt` → `calculateFrequencyScore()`

---

### 4. Color Space Analysis

**What is it?**
Analyzes if skin tones fall within natural human ranges.

**Why use it?**
Different materials (paper, plastic, screen) have different color spectrums.

**How it works:**
```
1. Convert RGB → YCrCb color space
   (Better for skin detection than RGB)

2. Define real skin ranges:
   Y: 0-255
   Cr: 133-173
   Cb: 77-127

3. Count pixels in range
4. Real faces: 60-90% pixels in range
   Fake attacks: <60% or >95% (too perfect)
```

**What it detects:**
- ✅ Paper (different reflectance)
- ✅ Plastic masks
- ✅ Screen color shift
- ❌ High-quality makeup

**Speed:** 2-3ms per frame

**Code Location:** `TraditionalCVAnalyzer.kt` → `calculateColorScore()`

---

### 5. Edge Sharpness (Laplacian)

**What is it?**
Measures image blur/sharpness using edge detection.

**Why use it?**
- Printed photos: Overly sharp (printer dots)
- Screen replays: Slightly blurred (pixel rendering)
- Real faces: Natural sharpness

**How it works:**
```
1. Apply Laplacian filter (edge detector)
2. Calculate variance of result
3. High variance = very sharp = printed
4. Low variance = blurry = screen
5. Medium variance = natural = real
```

**What it detects:**
- ✅ Printed photos (too sharp)
- ✅ Low-resolution screens (blurry)
- ✅ Out-of-focus attacks

**Speed:** 2-3ms per frame

**Code Location:** `TraditionalCVAnalyzer.kt` → `calculateTextureScore()`

---

### 6. Optical Flow (Motion Analysis)

**What is it?**
Tracks pixel movement between consecutive frames.

**Why use it?**
Video replays have unnaturally smooth motion (no micro-movements).

**How it works:**
```
1. Take current frame (gray)
2. Take previous frame (gray)
3. Calculate optical flow using Farneback algorithm
4. Get flow vectors for each pixel
5. Calculate magnitude (how much movement)
6. Analyze variance over time:
   - Real face: moderate variance (natural micro-movements)
   - Video replay: low variance (too smooth)
   - Photo: zero variance (no movement)
```

**Flow Visualization:**
```
Real Face:
Frame 1 → Frame 2: Small random movements everywhere
Variance: Medium

Video Replay:
Frame 1 → Frame 2: Smooth uniform movement
Variance: Low (too consistent)

Photo:
Frame 1 → Frame 2: Zero movement
Variance: Zero
```

**What it detects:**
- ✅ Video replays (smooth motion)
- ✅ Static photos (no motion)
- ❌ Doesn't help with live video deepfakes

**Speed:** 10-15ms per frame

**Code Location:** `MotionAnalyzer.kt`

---

### 7. Depth Analysis (Euler Angles)

**What is it?**
Tracks 3D head pose (yaw, pitch, roll) over time.

**Why use it?**
2D attacks (photos, screens) can't produce natural 3D head movements.

**How it works:**
```
1. ML Kit detects face landmarks
2. Calculate 3D head pose:
   - Yaw: Left-right rotation
   - Pitch: Up-down rotation
   - Roll: Tilt rotation

3. Track variance over 2-3 seconds:
   - Real face: Natural variance (5-50 degrees)
   - 2D attack: Low variance (<5 degrees)
```

**Euler Angles Explained:**
```
Yaw (Y-axis):
  -30° ← [Face] → +30°
  (Shaking head "no")

Pitch (X-axis):
  +30° ↑
  [Face]
  -30° ↓
  (Nodding "yes")

Roll (Z-axis):
  +15° ↻ [Face] ↺ -15°
  (Tilting head)
```

**What it detects:**
- ✅ Printed photos (no depth)
- ✅ Screen replays (flat)
- ✅ 2D masks

**Speed:** <5ms per frame (uses existing landmarks)

**Code Location:** `DepthAnalyzer.kt`

---

### 8. Blink Detection (EAR - Eye Aspect Ratio)

**What is it?**
Measures eye openness to detect blinks.

**Why use it?**
Final fallback that's nearly impossible to spoof with masks.

**How it works:**
```
1. Get 6 eye landmarks per eye
   (top, bottom, left, right, 2 intermediate)

2. Calculate EAR:
   EAR = (vertical_dist_1 + vertical_dist_2) / (2 × horizontal_dist)

3. Open eye: EAR ≈ 0.25-0.35
   Closed eye: EAR < 0.20

4. Detect blink:
   - EAR drops below 0.20 → Eye closed
   - EAR rises above 0.25 → Eye opened
   - Duration: 100-400ms → Valid blink

5. Count 2 natural blinks → PASS
```

**EAR Visualization:**
```
Open Eye:
  ___
 /   \   EAR = 0.30
 \___/

Half-Closed:
  ___
 /___\   EAR = 0.15

Closed:
 _____   EAR = 0.05
```

**Natural Blink Characteristics:**
- Duration: 100-400ms (too fast/slow = suspicious)
- Both eyes close simultaneously (±50ms)
- Smooth EAR curve (not abrupt)

**What it detects:**
- ✅ All static attacks (photos, screens, 2D masks)
- ✅ Most 3D masks (can't blink realistically)
- ❌ High-tech animatronic masks (rare)

**Speed:** 2-5ms per frame

**Code Location:** `BlinkDetector.kt`

---

### 9. Fusion Scorer

**What is it?**
Combines all detection scores into a single decision.

**Why weighted combination?**
Not all methods are equally reliable.

**Weighting Formula:**
```
final_score = 
    0.35 × cnn_score +           // CNN is primary
    0.15 × lbp_score +            // Good for prints
    0.10 × frequency_score +      // Specific to screens
    0.10 × color_score +          // General material check
    0.15 × motion_score +         // Catches videos
    0.10 × texture_score +        // Helps with quality
    0.05 × depth_score            // 3D validation

Thresholds:
- final_score ≥ 0.85 → PASS (High confidence real)
- final_score ≤ 0.55 → FAIL (High confidence fake)
- 0.55 < score < 0.85 → UNCERTAIN (Need blink test)
```

**Why these weights?**
- CNN gets highest (35%): Most comprehensive training
- Motion (15%): Critical for video detection
- LBP (15%): Fast and effective for photos
- Color/Frequency/Texture (10% each): Supplementary
- Depth (5%): Useful but can be noisy

**Attack Detection Rules:**
```kotlin
if (lbp_score < 0.3 AND motion_score < 0.3):
    attack_type = PRINTED_PHOTO

if (frequency_score < 0.3):
    attack_type = SCREEN_REPLAY

if (depth_score < 0.3 AND color_score < 0.4):
    attack_type = MASK_2D

if (motion_score < 0.4 AND cnn_score > 0.6):
    attack_type = VIDEO_REPLAY
```

**Code Location:** `FusionScorer.kt`

---

## Complete Code Flow

### Application Startup

```
1. MainActivity.onCreate()
   ↓
2. Check camera permission
   ↓
3. Initialize OpenCV library
   ↓
4. Initialize ML Kit (Face Detection + Face Mesh)
   ↓
5. Load TFLite model (Silent-Face)
   ↓
6. Initialize all analyzers:
   - CNNInference
   - TraditionalCVAnalyzer
   - MotionAnalyzer
   - DepthAnalyzer
   - BlinkDetector
   - FusionScorer
   ↓
7. Start camera preview
   ↓
8. Set stage = INITIALIZING
```

### Frame Processing Loop

```
Every camera frame (10-20 FPS):

1. CameraPreview.FaceAnalyzer.analyze()
   ↓
2. ML Kit Face Detection
   ├─ No face → Skip frame
   └─ Face found → Continue
   ↓
3. Extract data:
   ├─ Face bitmap (crop from frame)
   ├─ 468 facial landmarks
   └─ Euler angles (yaw, pitch, roll)
   ↓
4. Pass to ViewModel.processFrame()
   ↓
5. ViewModel calls LivenessDetector.processFrame()
   ↓
6. LivenessDetector routes based on current stage:
   ├─ INITIALIZING → Start passive analysis
   ├─ PASSIVE_ANALYSIS → Process passive frame
   ├─ ACTIVE_PROMPT → Process blink detection
   └─ COMPLETED → Do nothing (awaiting reset)
```

### Passive Analysis (Stage 1)

```
For 25 frames (2-3 seconds):

1. LivenessDetector.processPassiveFrame()
   ↓
2. Run all analyses in PARALLEL:
   ├─ CNNInference.predictLiveness() → 40-60ms
   ├─ TraditionalCVAnalyzer.calculateLBPScore() → 3-5ms
   ├─ TraditionalCVAnalyzer.calculateFrequencyScore() → 5-8ms
   ├─ TraditionalCVAnalyzer.calculateColorScore() → 2-3ms
   ├─ TraditionalCVAnalyzer.calculateTextureScore() → 2-3ms
   ├─ MotionAnalyzer.analyzeMotion() → 10-15ms
   └─ DepthAnalyzer.analyzeDepth() → <5ms
   ↓
3. FusionScorer.calculateFinalScore()
   ↓
4. Store scores in buffer
   ↓
5. Update UI:
   ├─ Progress bar (frame X/25)
   ├─ Instruction text
   └─ Bounding box color (yellow)
   ↓
6. After 25 frames → makePassiveDecision()
```

### Decision Making

```
LivenessDetector.makePassiveDecision():

1. Aggregate 25 frame scores using MEDIAN
   (Median is robust against outliers)
   ↓
2. Calculate final_score
   ↓
3. Apply thresholds:
   ├─ ≥ 0.85 → HIGH CONFIDENCE LIVE
   │   ├─ Create LivenessResult(isPassed=true)
   │   ├─ Set stage = COMPLETED
   │   └─ Show green PASS banner
   │
   ├─ ≤ 0.55 → HIGH CONFIDENCE SPOOF
   │   ├─ Detect attack type
   │   ├─ Create LivenessResult(isPassed=false)
   │   ├─ Set stage = COMPLETED
   │   └─ Show red FAIL banner
   │
   └─ 0.55-0.85 → UNCERTAIN
       ├─ Reset BlinkDetector
       ├─ Set stage = ACTIVE_PROMPT
       └─ Show "Please blink twice" instruction
```

### Active Analysis (Stage 2)

```
If passive was uncertain:

For each frame (until 2 blinks or timeout):

1. LivenessDetector.processActiveFrame()
   ↓
2. BlinkDetector.detectBlink(landmarks)
   ↓
3. Calculate EAR for both eyes
   ↓
4. Detect blink state changes:
   ├─ EAR drops < 0.20 → Eye closing
   ├─ EAR rises > 0.25 → Eye opening
   └─ Duration 100-400ms → Valid blink
   ↓
5. Update UI:
   ├─ Blink count display
   ├─ Instruction text
   └─ Bounding box (yellow)
   ↓
6. If 2 valid blinks → makeActiveDecision()
   ↓
7. Validate blink naturalness:
   ├─ Timing correct?
   ├─ Both eyes?
   └─ Smooth motion?
   ↓
8. Create final result:
   ├─ Natural blinks → PASS
   └─ Unnatural/no blinks → FAIL
```

### UI Updates

```
ViewModel observes stage changes:

1. LivenessDetector emits new stage
   ↓
2. ViewModel updates StateFlows:
   ├─ stage
   ├─ instructionText
   └─ faceBox (bounding box)
   ↓
3. Compose UI reacts:
   ├─ StatusPanel shows current stage
   ├─ Progress bars update
   ├─ Instruction text changes
   ├─ Bounding box color changes
   └─ Result banner appears (if completed)
   ↓
4. User sees real-time feedback
```

### Reset Flow

```
User taps DONE/RETRY:

1. LivenessViewModel.reset()
   ↓
2. LivenessDetector.reset()
   ├─ Clear frame scores
   ├─ Reset MotionAnalyzer
   ├─ Reset DepthAnalyzer
   ├─ Reset BlinkDetector
   └─ Set stage = INITIALIZING
   ↓
3. UI resets:
   ├─ Clear bounding box
   ├─ Reset instruction text
   └─ Hide result banner
   ↓
4. Ready for next detection
```

---

## Step-by-Step Implementation Guide

### Phase 1: Project Setup (30 minutes)

**Step 1.1: Create Android Project**
```
1. Open Android Studio
2. New Project → Empty Compose Activity
3. Name: LivenessDetection
4. Package: com.uidai.livenessdetection
5. Minimum SDK: API 26 (Android 8.0)
6. Language: Kotlin
```

**Step 1.2: Update build.gradle Files**
```
1. Copy project-level build.gradle
2. Copy app-level build.gradle
3. Sync project
4. Verify all dependencies download
```

**Step 1.3: Add Permissions**
```
1. Open AndroidManifest.xml
2. Add camera permission
3. Add camera hardware requirement
4. Set screen orientation to portrait
```

---

### Phase 2: Data Models (20 minutes)

**Step 2.1: Create Package Structure**
```
app/src/main/java/com/uidai/livenessdetection/
├── domain/
│   ├── LivenessDetector.kt
│   └── models/
│       └── LivenessResult.kt
├── ml/
│   ├── CNNInference.kt
│   ├── TraditionalCVAnalyzer.kt
│   ├── MotionAnalyzer.kt
│   ├── BlinkDetector.kt
│   └── FusionScorer.kt
└── ui/
    ├── MainActivity.kt
    ├── LivenessDetectionScreen.kt
    ├── CameraPreview.kt
    └── LivenessViewModel.kt
```

**Step 2.2: Create Data Classes**
```
1. Copy LivenessResult.kt code
2. Create all sealed classes and data classes
3. Verify no compilation errors
```

---

### Phase 3: ML Components (2 hours)

**Step 3.1: CNN Inference**
```
1. Create CNNInference.kt
2. Implement model loading
3. Implement preprocessing
4. Implement inference
5. Test with dummy bitmap
```

**Step 3.2: Traditional CV**
```
1. Initialize OpenCV in MainActivity
2. Create TraditionalCVAnalyzer.kt
3. Implement LBP calculation
4. Implement Fourier analysis
5. Implement color analysis
6. Implement texture analysis
7. Test each method individually
```

**Step 3.3: Motion & Depth**
```
1. Create MotionAnalyzer.kt
2. Implement optical flow
3. Create DepthAnalyzer.kt
4. Implement euler angle tracking
5. Test with sample frames
```

**Step 3.4: Blink Detection**
```
1. Create BlinkDetector.kt
2. Implement EAR calculation
3. Implement blink state machine
4. Test with eye landmark data
```

**Step 3.5: Fusion Scorer**
```
1. Create FusionScorer.kt
2. Implement weighted scoring
3. Implement decision thresholds
4. Implement attack type detection
```

---

### Phase 4: Core Logic (1 hour)

**Step 4.1: Liveness Detector**
```
1. Create LivenessDetector.kt
2. Initialize all analyzers
3. Implement processFrame()
4. Implement passive analysis
5. Implement active analysis
6. Implement decision logic
7. Implement aggregation
```

---

### Phase 5: Camera Integration (1 hour)

**Step 5.1: Camera Preview**
```
1. Add CameraX dependencies
2. Create CameraPreview.kt
3. Implement face detection
4. Implement face mesh detection
5. Extract landmarks and angles
6. Create callback to ViewModel
```

---

### Phase 6: UI Implementation (2 hours)

**Step 6.1: ViewModel**
```
1. Create LivenessViewModel.kt
2. Implement StateFlows
3. Implement frame processing
4. Implement reset logic
```

**Step 6.2: Compose UI**
```
1. Create MainActivity.kt
2. Handle camera permission
3. Create LivenessDetectionScreen.kt
4. Implement camera preview section
5. Implement status panels
6. Implement result banner
7. Add bounding box overlay
```

---

### Phase 7: Model Acquisition (