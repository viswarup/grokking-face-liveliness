package com.uidai.livenessdetection.ml

import android.graphics.PointF
import android.util.Log
import kotlin.math.sqrt

/**
 * Blink detector using ML Kit's Eye Open Probability.
 * Detects natural blinks by monitoring eye open probability over time.
 */
class BlinkDetector {
    
    companion object {
        private const val TAG = "BlinkDetector"
        const val REQUIRED_BLINKS = 2
        
        // Eye open probability thresholds
        private const val EYE_CLOSED_THRESHOLD = 0.3f    // Below this = eyes closed
        private const val EYE_OPEN_THRESHOLD = 0.7f      // Above this = eyes open
        private const val MIN_BLINK_DURATION_MS = 50L
        private const val MAX_BLINK_DURATION_MS = 500L
    }
    
    private var blinkCount = 0
    private var isEyeClosed = false
    private var eyeClosedStartTime = 0L
    private val blinkTimings = mutableListOf<Long>()
    private var lastLogTime = 0L
    
    /**
     * Detects blinks using eye open probability from ML Kit.
     * 
     * @param landmarks List of face landmarks. The last element (index 6) contains
     *                  eye probabilities as PointF(leftEyeOpenProb, rightEyeOpenProb)
     * @return BlinkResult with detection status
     */
    fun detectBlink(landmarks: List<PointF>): BlinkResult {
        // Get eye open probabilities from landmarks
        // They're stored at the last index as PointF(leftProb, rightProb)
        val eyeProbs = if (landmarks.size > 6) landmarks[6] else null
        
        if (eyeProbs == null) {
            Log.d(TAG, "No eye probability data available")
            return BlinkResult(
                blinkCount = blinkCount,
                isComplete = blinkCount >= REQUIRED_BLINKS,
                isNatural = true,
                currentEAR = 0.5f
            )
        }
        
        val leftEyeOpen = eyeProbs.x
        val rightEyeOpen = eyeProbs.y
        val avgEyeOpen = (leftEyeOpen + rightEyeOpen) / 2f
        
        // Log periodically (not every frame)
        val now = System.currentTimeMillis()
        if (now - lastLogTime > 500) {
            Log.d(TAG, "Eye open probability: L=$leftEyeOpen, R=$rightEyeOpen, avg=$avgEyeOpen")
            lastLogTime = now
        }
        
        // Detect blink state transition
        val eyesClosed = avgEyeOpen < EYE_CLOSED_THRESHOLD
        val eyesOpen = avgEyeOpen > EYE_OPEN_THRESHOLD
        
        if (eyesClosed && !isEyeClosed) {
            // Transition: Eyes just closed
            isEyeClosed = true
            eyeClosedStartTime = now
            Log.d(TAG, "👁 Eyes CLOSED detected")
        } else if (eyesOpen && isEyeClosed) {
            // Transition: Eyes just opened - potential blink completed
            val blinkDuration = now - eyeClosedStartTime
            
            if (blinkDuration in MIN_BLINK_DURATION_MS..MAX_BLINK_DURATION_MS) {
                blinkCount++
                blinkTimings.add(blinkDuration)
                Log.d(TAG, "✅ BLINK DETECTED! Count: $blinkCount, Duration: ${blinkDuration}ms")
            } else {
                Log.d(TAG, "⚠ Eye closure too ${if (blinkDuration < MIN_BLINK_DURATION_MS) "short" else "long"}: ${blinkDuration}ms")
            }
            
            isEyeClosed = false
        }
        
        val isComplete = blinkCount >= REQUIRED_BLINKS
        val isNatural = validateNaturalBlinks()
        
        if (isComplete) {
            Log.d(TAG, "🎉 Required blinks completed! Natural: $isNatural")
        }
        
        return BlinkResult(
            blinkCount = blinkCount,
            isComplete = isComplete,
            isNatural = isNatural,
            currentEAR = avgEyeOpen
        )
    }
    
    /**
     * Validates that detected blinks appear natural.
     * Checks for timing variance (natural blinks have some variance).
     */
    private fun validateNaturalBlinks(): Boolean {
        if (blinkTimings.size < 2) return true
        
        val avgDuration = blinkTimings.average()
        val variance = blinkTimings.map { (it - avgDuration) * (it - avgDuration) }.average()
        val stdDev = sqrt(variance)
        
        // Natural blinks have some variance (10-150ms std dev)
        // If stdDev is too low (exactly same timing), might be video replay
        val isNatural = stdDev in 5.0..200.0 || blinkTimings.size < 3
        
        Log.d(TAG, "Blink timing analysis: avg=${avgDuration}ms, stdDev=${stdDev}ms, natural=$isNatural")
        
        return isNatural
    }
    
    /**
     * Resets the detector for a new session.
     */
    fun reset() {
        blinkCount = 0
        isEyeClosed = false
        eyeClosedStartTime = 0L
        blinkTimings.clear()
        lastLogTime = 0L
        Log.d(TAG, "BlinkDetector reset")
    }
}

/**
 * Result of blink detection.
 */
data class BlinkResult(
    val blinkCount: Int,
    val isComplete: Boolean,
    val isNatural: Boolean,
    val currentEAR: Float
)
