package com.uidai.livenessdetection.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log

/**
 * CNN-based liveness detection wrapper.
 * Now uses ONNX Runtime with MiniFASNet models for inference.
 * 
 * This class provides backward compatibility with the existing pipeline
 * while using the new ONNX-based detector internally.
 */
class CNNInference(private val context: Context) {
    
    companion object {
        private const val TAG = "CNNInference"
    }
    
    private val onnxDetector = OnnxLivenessDetector(context)
    private var lastFaceBox: Rect? = null
    
    /**
     * Update the face bounding box for prediction.
     * This should be called before predictLiveness() with the current face box.
     */
    fun setFaceBox(box: Rect) {
        lastFaceBox = box
    }
    
    /**
     * Predicts whether the face is live or spoofed.
     * @param faceBitmap Full camera frame (not just cropped face)
     * @return Probability of being a real face (0.0 to 1.0)
     */
    fun predictLiveness(faceBitmap: Bitmap): Float {
        val faceBox = lastFaceBox
        
        if (faceBox == null) {
            Log.d(TAG, "No face box available, using center crop")
            // Use center region as fallback
            val size = minOf(faceBitmap.width, faceBitmap.height) / 2
            val centerX = faceBitmap.width / 2
            val centerY = faceBitmap.height / 2
            lastFaceBox = Rect(
                centerX - size / 2,
                centerY - size / 2,
                centerX + size / 2,
                centerY + size / 2
            )
        }
        
        return try {
            val result = onnxDetector.predict(faceBitmap, lastFaceBox!!)
            Log.d(TAG, "ONNX prediction: isReal=${result.isReal}, realScore=${result.realScore}")
            result.realScore
        } catch (e: Exception) {
            Log.e(TAG, "Prediction error: ${e.message}")
            0.5f  // Return neutral score on error
        }
    }
    
    /**
     * Releases the interpreter resources.
     */
    fun close() {
        onnxDetector.close()
        Log.d(TAG, "CNNInference closed")
    }
}
