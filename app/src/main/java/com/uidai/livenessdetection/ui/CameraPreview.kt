package com.uidai.livenessdetection.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * Camera preview composable that captures frames and performs face detection.
 * Uses ML Kit Face Detection for face tracking and landmark detection.
 */
@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onFrameAnalyzed: (Bitmap, List<PointF>, FloatArray, Rect, Int, Int) -> Unit,
    onNoFaceDetected: () -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    
    val faceDetector = remember {
        try {
            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)  // Needed for eye open probability
                .setMinFaceSize(0.15f)
                .build()
            FaceDetection.getClient(options)
        } catch (e: Exception) {
            Log.e("CameraPreview", "Failed to create face detector", e)
            null
        }
    }
    
    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            faceDetector?.close()
        }
    }
    
    AndroidView(
        modifier = modifier.fillMaxSize(),
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    
                    val preview = Preview.Builder()
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                    
                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            if (faceDetector != null) {
                                it.setAnalyzer(
                                    cameraExecutor,
                                    FaceAnalyzer(
                                        faceDetector,
                                        onFrameAnalyzed,
                                        onNoFaceDetected
                                    )
                                )
                            }
                        }
                    
                    val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                    
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )
                    
                    Log.d("CameraPreview", "Camera bound successfully")
                } catch (exc: Exception) {
                    Log.e("CameraPreview", "Camera binding failed", exc)
                }
            }, ContextCompat.getMainExecutor(ctx))
            
            previewView
        }
    )
}

/**
 * Image analyzer that performs face detection on camera frames.
 * Extracts face landmarks, bounding box, Euler angles, and eye probabilities.
 */
private class FaceAnalyzer(
    private val faceDetector: FaceDetector,
    private val onFrameAnalyzed: (Bitmap, List<PointF>, FloatArray, Rect, Int, Int) -> Unit,
    private val onNoFaceDetected: () -> Unit
) : ImageAnalysis.Analyzer {
    
    companion object {
        private const val TAG = "FaceAnalyzer"
    }
    
    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        try {
            val mediaImage = imageProxy.image
            if (mediaImage == null) {
                imageProxy.close()
                return
            }
            
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            
            val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)
            
            faceDetector.process(image)
                .addOnSuccessListener { faces ->
                    try {
                        if (faces.isNotEmpty()) {
                            val face = faces[0]
                            processFace(face, imageProxy, imageWidth, imageHeight, rotationDegrees)
                        } else {
                            onNoFaceDetected()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing face: ${e.message}")
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed: ${e.message}")
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error in analyze: ${e.message}")
            imageProxy.close()
        }
    }
    
    /**
     * Processes detected face and extracts features.
     */
    private fun processFace(
        face: Face,
        imageProxy: ImageProxy,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ) {
        // Get Euler angles for head pose
        val eulerAngles = floatArrayOf(
            face.headEulerAngleY,  // Yaw (-45 to 45)
            face.headEulerAngleX,  // Pitch (-45 to 45)
            face.headEulerAngleZ   // Roll (-45 to 45)
        )
        
        // Get bounding box
        val boundingBox = face.boundingBox
        
        // Build landmarks list with eye open probabilities
        val landmarks = mutableListOf<PointF>()
        
        // Add standard landmarks
        face.getLandmark(FaceLandmark.LEFT_EYE)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        face.getLandmark(FaceLandmark.RIGHT_EYE)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        face.getLandmark(FaceLandmark.NOSE_BASE)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        face.getLandmark(FaceLandmark.MOUTH_LEFT)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        face.getLandmark(FaceLandmark.MOUTH_BOTTOM)?.let {
            landmarks.add(PointF(it.position.x, it.position.y))
        }
        
        // Store eye open probabilities in landmarks list (indices 6 and 7)
        // Using PointF to pass as (leftEyeOpenProb, rightEyeOpenProb)
        val leftEyeOpen = face.leftEyeOpenProbability ?: 0.5f
        val rightEyeOpen = face.rightEyeOpenProbability ?: 0.5f
        landmarks.add(PointF(leftEyeOpen, rightEyeOpen))  // Index 6: eye probabilities
        
        Log.d(TAG, "Face detected: box=${boundingBox}, euler=[${eulerAngles[0]}, ${eulerAngles[1]}, ${eulerAngles[2]}], " +
                "eyeOpen=[L:$leftEyeOpen, R:$rightEyeOpen]")
        
        // Convert to bitmap safely
        val bitmap = imageProxyToBitmap(imageProxy)
        if (bitmap != null && landmarks.isNotEmpty()) {
            // Pass image dimensions for proper scaling
            onFrameAnalyzed(bitmap, landmarks, eulerAngles, boundingBox, imageWidth, imageHeight)
        }
    }
    
    /**
     * Converts ImageProxy to Bitmap using multiple methods.
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            imageProxy.toBitmap()
        } catch (e: Exception) {
            Log.d(TAG, "toBitmap() failed, trying YUV conversion")
            try {
                yuvToRgb(imageProxy)
            } catch (e2: Exception) {
                Log.e(TAG, "All bitmap conversions failed: ${e2.message}")
                null
            }
        }
    }
    
    /**
     * Converts YUV ImageProxy to RGB Bitmap.
     */
    private fun yuvToRgb(imageProxy: ImageProxy): Bitmap? {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val imageBytes = out.toByteArray()
        
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            matrix.postScale(-1f, 1f) // Mirror for front camera
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }
}
