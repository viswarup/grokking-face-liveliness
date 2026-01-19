package com.uidai.livenessdetection.ml

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Analyzes motion between frames to detect video replay attacks.
 * Uses simple frame differencing instead of OpenCV optical flow.
 */
class MotionAnalyzer {
    
    companion object {
        private const val TAG = "MotionAnalyzer"
        private const val HISTORY_SIZE = 10
    }
    
    private val motionHistory = mutableListOf<Float>()
    private var previousGrayPixels: IntArray? = null
    private var previousWidth = 0
    private var previousHeight = 0
    
    /**
     * Analyzes motion between current and previous frame.
     * @param currentBitmap Current camera frame
     * @return Score between 0.0 (likely video replay) and 1.0 (likely real)
     */
    fun analyzeMotion(currentBitmap: Bitmap): Float {
        return try {
            val width = currentBitmap.width
            val height = currentBitmap.height
            val currentGray = bitmapToGrayscale(currentBitmap)
            
            val motionMagnitude: Float
            
            if (previousGrayPixels != null && 
                previousWidth == width && 
                previousHeight == height) {
                
                // Calculate frame difference
                var diffSum = 0L
                var diffCount = 0
                val diffValues = mutableListOf<Int>()
                
                // Sample pixels for performance
                val step = 4
                for (y in 0 until height step step) {
                    for (x in 0 until width step step) {
                        val idx = y * width + x
                        val diff = abs(currentGray[idx] - previousGrayPixels!![idx])
                        diffSum += diff
                        diffValues.add(diff)
                        diffCount++
                    }
                }
                
                val meanDiff = diffSum.toFloat() / diffCount
                
                // Calculate variance of differences
                val variance = diffValues.map { (it - meanDiff) * (it - meanDiff) }
                    .average().toFloat()
                
                motionMagnitude = sqrt(variance)
            } else {
                motionMagnitude = 10f // Default for first frame
            }
            
            // Store for next frame
            previousGrayPixels = currentGray
            previousWidth = width
            previousHeight = height
            
            // Add to history
            motionHistory.add(motionMagnitude)
            if (motionHistory.size > HISTORY_SIZE) {
                motionHistory.removeAt(0)
            }
            
            calculateMotionScore()
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing motion: ${e.message}")
            0.5f
        }
    }
    
    /**
     * Calculates motion score based on motion history.
     * Real faces have natural, variable motion; videos have smoother patterns.
     */
    private fun calculateMotionScore(): Float {
        if (motionHistory.size < 3) return 0.5f
        
        val mean = motionHistory.average().toFloat()
        val variance = motionHistory.map { (it - mean) * (it - mean) }.average().toFloat()
        val stdDev = sqrt(variance)
        
        // Real faces: moderate motion with some variance
        // Video replay: very smooth or periodic motion
        // Photo: very low motion
        
        val score = when {
            mean < 2f && stdDev < 1f -> 0.3f  // Too still (photo)
            mean > 50f -> 0.4f                 // Too much motion (shaking phone)
            stdDev < 2f && mean > 5f -> 0.4f  // Too smooth (video)
            else -> 0.6f + (stdDev.coerceIn(2f, 15f) - 2f) / 13f * 0.3f
        }
        
        Log.d(TAG, "Motion score: $score (mean: $mean, stdDev: $stdDev)")
        return score.coerceIn(0f, 1f)
    }
    
    /**
     * Resets the analyzer state.
     */
    fun reset() {
        motionHistory.clear()
        previousGrayPixels = null
        previousWidth = 0
        previousHeight = 0
    }
    
    /**
     * Converts bitmap to grayscale array.
     */
    private fun bitmapToGrayscale(bitmap: Bitmap): IntArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        val grayPixels = IntArray(width * height)
        
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            grayPixels[i] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }
        
        return grayPixels
    }
}

/**
 * Analyzes depth cues from head pose variation.
 * 2D attacks (photos, screens) show less natural head pose variation.
 */
class DepthAnalyzer {
    
    companion object {
        private const val TAG = "DepthAnalyzer"
        private const val HISTORY_SIZE = 20
    }
    
    private val eulerHistory = mutableListOf<FloatArray>()
    
    /**
     * Analyzes head pose variation for depth estimation.
     * @param eulerAngles [yaw, pitch, roll] from face detection
     * @return Score between 0.0 (likely 2D attack) and 1.0 (likely real 3D face)
     */
    fun analyzeDepth(eulerAngles: FloatArray): Float {
        return try {
            // Store Euler angles
            eulerHistory.add(eulerAngles.copyOf())
            if (eulerHistory.size > HISTORY_SIZE) {
                eulerHistory.removeAt(0)
            }
            
            if (eulerHistory.size < 5) return 0.5f
            
            // Calculate variance for each angle
            val yawVariance = calculateVariance(eulerHistory.map { it[0] })
            val pitchVariance = calculateVariance(eulerHistory.map { it[1] })
            val rollVariance = calculateVariance(eulerHistory.map { it[2] })
            
            // Real faces have natural head movements
            // 2D photos have very low variance (fixed position)
            // Videos might have smooth, periodic motion
            
            val totalVariance = yawVariance + pitchVariance + rollVariance
            
            val score = when {
                totalVariance < 1f -> 0.3f      // Too still
                totalVariance > 500f -> 0.4f   // Too erratic
                else -> 0.5f + (totalVariance.coerceIn(1f, 50f) - 1f) / 49f * 0.4f
            }
            
            Log.d(TAG, "Depth score: $score (variance: $totalVariance)")
            score.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing depth: ${e.message}")
            0.5f
        }
    }
    
    /**
     * Calculates variance of a list of values.
     */
    private fun calculateVariance(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average().toFloat()
        return values.map { (it - mean) * (it - mean) }.average().toFloat()
    }
    
    /**
     * Resets the analyzer state.
     */
    fun reset() {
        eulerHistory.clear()
    }
}
