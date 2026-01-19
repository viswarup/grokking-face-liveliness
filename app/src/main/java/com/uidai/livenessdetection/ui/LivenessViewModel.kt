package com.uidai.livenessdetection.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.uidai.livenessdetection.domain.LivenessDetector
import com.uidai.livenessdetection.domain.models.LivenessStage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for liveness detection screen.
 * Manages UI state and coordinates with the LivenessDetector.
 */
class LivenessViewModel(application: Application) : AndroidViewModel(application) {
    
    companion object {
        private const val TAG = "LivenessViewModel"
    }
    
    private val livenessDetector = LivenessDetector(application)
    
    private val _stage = MutableStateFlow<LivenessStage>(LivenessStage.Initializing)
    val stage: StateFlow<LivenessStage> = _stage.asStateFlow()
    
    private val _faceBox = MutableStateFlow<Rect?>(null)
    val faceBox: StateFlow<Rect?> = _faceBox.asStateFlow()
    
    // Store image dimensions for proper scaling
    private val _imageDimensions = MutableStateFlow(Pair(640, 480))
    val imageDimensions: StateFlow<Pair<Int, Int>> = _imageDimensions.asStateFlow()
    
    private val _instructionText = MutableStateFlow("Position your face in the frame")
    val instructionText: StateFlow<String> = _instructionText.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    // Current face bounding box for ONNX inference
    private var currentFaceBox: Rect? = null
    
    /**
     * Processes a camera frame with face data.
     * Now includes face bounding box for ONNX preprocessing.
     */
    fun processFrame(
        bitmap: Bitmap, 
        landmarks: List<PointF>, 
        eulerAngles: FloatArray,
        imageWidth: Int,
        imageHeight: Int
    ) {
        if (_isProcessing.value) return
        
        // Update image dimensions
        _imageDimensions.value = Pair(imageWidth, imageHeight)
        
        viewModelScope.launch {
            _isProcessing.value = true
            try {
                // Pass current face box to detector for ONNX preprocessing
                val newStage = livenessDetector.processFrame(
                    bitmap, 
                    landmarks, 
                    eulerAngles,
                    currentFaceBox
                )
                _stage.value = newStage
                updateInstructionText(newStage)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}")
            } finally {
                _isProcessing.value = false
            }
        }
    }
    
    /**
     * Updates the face bounding box for UI overlay and ONNX inference.
     */
    fun updateFaceBox(rect: Rect?) {
        _faceBox.value = rect
        currentFaceBox = rect
    }
    
    /**
     * Updates instruction text based on current stage.
     */
    private fun updateInstructionText(stage: LivenessStage) {
        _instructionText.value = when (stage) {
            is LivenessStage.Initializing -> "Initializing camera..."
            is LivenessStage.FaceAlignment -> "Center your face in the frame"
            is LivenessStage.PassiveAnalysis -> "Hold steady, analyzing... (${stage.frameCount}/${stage.totalFrames})"
            is LivenessStage.ActivePrompt -> "Please blink twice naturally (${stage.blinkCount}/${stage.requiredBlinks})"
            is LivenessStage.Completed -> {
                if (stage.result.isPassed) "✓ Verification Complete!"
                else "✗ Verification Failed"
            }
        }
    }
    
    /**
     * Resets the detection for a new session.
     */
    fun reset() {
        livenessDetector.reset()
        _stage.value = LivenessStage.Initializing
        _instructionText.value = "Position your face in the frame"
        _faceBox.value = null
        currentFaceBox = null
        Log.d(TAG, "ViewModel reset")
    }
    
    override fun onCleared() {
        super.onCleared()
        livenessDetector.release()
        Log.d(TAG, "ViewModel cleared")
    }
}
