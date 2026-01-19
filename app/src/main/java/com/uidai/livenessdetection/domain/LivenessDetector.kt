package com.uidai.livenessdetection.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import android.util.Log
import com.uidai.livenessdetection.domain.models.*
import com.uidai.livenessdetection.ml.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Main orchestrator for liveness detection.
 * Coordinates all detection modules and manages the detection flow.
 */
class LivenessDetector(context: Context) {
    
    companion object {
        private const val TAG = "LivenessDetector"
        private const val REQUIRED_FRAMES = 10  // Reduced from 25 for faster detection
    }
    
    private val cnnInference = CNNInference(context)
    private val cvAnalyzer = TraditionalCVAnalyzer()
    private val motionAnalyzer = MotionAnalyzer()
    private val depthAnalyzer = DepthAnalyzer()
    private val blinkDetector = BlinkDetector()
    private val fusionScorer = FusionScorer()
    
    private val frameScores = mutableListOf<PassiveScores>()
    private var currentStage: LivenessStage = LivenessStage.Initializing
    
    /**
     * Processes a single camera frame.
     * @param faceBitmap Full camera frame bitmap
     * @param landmarks List of face landmarks (last element contains eye probabilities)
     * @param eulerAngles Head pose [yaw, pitch, roll] in degrees
     * @param faceBox Bounding box of detected face
     * @return Current detection stage
     */
    suspend fun processFrame(
        faceBitmap: Bitmap,
        landmarks: List<PointF>,
        eulerAngles: FloatArray,
        faceBox: Rect? = null
    ): LivenessStage = withContext(Dispatchers.Default) {
        
        // Pass face box to CNN for proper cropping
        faceBox?.let { cnnInference.setFaceBox(it) }
        
        when (currentStage) {
            is LivenessStage.Initializing,
            is LivenessStage.FaceAlignment -> {
                currentStage = LivenessStage.PassiveAnalysis(0, REQUIRED_FRAMES)
                Log.d(TAG, "Starting passive analysis")
                return@withContext currentStage
            }
            
            is LivenessStage.PassiveAnalysis -> {
                return@withContext processPassiveFrame(faceBitmap, landmarks, eulerAngles)
            }
            
            is LivenessStage.ActivePrompt -> {
                return@withContext processActiveFrame(landmarks)
            }
            
            else -> return@withContext currentStage
        }
    }
    
    /**
     * Processes a frame during passive analysis stage.
     */
    private suspend fun processPassiveFrame(
        faceBitmap: Bitmap,
        landmarks: List<PointF>,
        eulerAngles: FloatArray
    ): LivenessStage = withContext(Dispatchers.Default) {
        
        try {
            // Run all analyses
            val cnnScore = cnnInference.predictLiveness(faceBitmap)
            val lbpScore = cvAnalyzer.calculateLBPScore(faceBitmap)
            val frequencyScore = cvAnalyzer.calculateFrequencyScore(faceBitmap)
            val colorScore = cvAnalyzer.calculateColorScore(faceBitmap)
            val textureScore = cvAnalyzer.calculateTextureScore(faceBitmap)
            val motionScore = motionAnalyzer.analyzeMotion(faceBitmap)
            val depthScore = depthAnalyzer.analyzeDepth(eulerAngles)
            
            // Calculate fusion score
            val scores = fusionScorer.calculateFinalScore(
                cnnScore, lbpScore, frequencyScore, colorScore,
                motionScore, textureScore, depthScore
            )
            
            frameScores.add(scores)
            
            // Update stage with current scores for UI display
            currentStage = LivenessStage.PassiveAnalysis(
                frameCount = frameScores.size,
                totalFrames = REQUIRED_FRAMES,
                currentScores = scores
            )
            
            Log.d(TAG, "Processed frame ${frameScores.size}/$REQUIRED_FRAMES, score: ${scores.finalScore}")
            
            // Check if we have enough frames
            if (frameScores.size >= REQUIRED_FRAMES) {
                return@withContext makePassiveDecision()
            }
            
            currentStage
        } catch (e: Exception) {
            Log.e(TAG, "Error processing passive frame: ${e.message}")
            currentStage
        }
    }
    
    /**
     * Makes the final decision after passive analysis.
     */
    private fun makePassiveDecision(): LivenessStage {
        // Aggregate scores using median (robust against outliers)
        val aggregatedScore = aggregateScores(frameScores)
        val decision = fusionScorer.getDecision(aggregatedScore.finalScore)
        
        Log.d(TAG, "Passive decision: $decision (final score: ${aggregatedScore.finalScore})")
        
        return when (decision) {
            FusionScorer.Decision.PASS -> {
                val result = LivenessResult(
                    isPassed = true,
                    confidence = aggregatedScore.finalScore,
                    passiveScore = aggregatedScore,
                    activeResult = null,
                    attackType = null
                )
                currentStage = LivenessStage.Completed(result)
                currentStage
            }
            
            FusionScorer.Decision.FAIL -> {
                val attackType = fusionScorer.detectAttackType(aggregatedScore)
                val result = LivenessResult(
                    isPassed = false,
                    confidence = 1.0f - aggregatedScore.finalScore,
                    passiveScore = aggregatedScore,
                    activeResult = null,
                    attackType = attackType
                )
                currentStage = LivenessStage.Completed(result)
                currentStage
            }
            
            FusionScorer.Decision.UNCERTAIN -> {
                // Trigger active liveness
                Log.d(TAG, "Passive uncertain, triggering active liveness")
                blinkDetector.reset()
                currentStage = LivenessStage.ActivePrompt(0, BlinkDetector.REQUIRED_BLINKS)
                currentStage
            }
        }
    }
    
    /**
     * Processes a frame during active (blink) detection stage.
     */
    private fun processActiveFrame(landmarks: List<PointF>): LivenessStage {
        val blinkResult = blinkDetector.detectBlink(landmarks)
        
        currentStage = LivenessStage.ActivePrompt(
            blinkCount = blinkResult.blinkCount,
            requiredBlinks = BlinkDetector.REQUIRED_BLINKS
        )
        
        if (blinkResult.isComplete) {
            return makeActiveDecision(blinkResult)
        }
        
        return currentStage
    }
    
    /**
     * Makes the final decision after active liveness check.
     */
    private fun makeActiveDecision(blinkResult: BlinkResult): LivenessStage {
        val aggregatedScore = if (frameScores.isNotEmpty()) {
            aggregateScores(frameScores)
        } else {
            PassiveScores(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f)
        }
        
        val isPassed = blinkResult.isNatural && blinkResult.isComplete
        
        Log.d(TAG, "Active decision: passed=$isPassed, natural=${blinkResult.isNatural}")
        
        val activeResult = ActiveResult(
            blinkDetected = blinkResult.isComplete,
            blinkCount = blinkResult.blinkCount,
            blinkTimings = emptyList(),
            isNatural = blinkResult.isNatural
        )
        
        val result = LivenessResult(
            isPassed = isPassed,
            confidence = if (isPassed) 0.90f else 0.85f,
            passiveScore = aggregatedScore,
            activeResult = activeResult,
            attackType = if (!isPassed) fusionScorer.detectAttackType(aggregatedScore) else null
        )
        
        currentStage = LivenessStage.Completed(result)
        return currentStage
    }
    
    /**
     * Aggregates multiple frame scores using median.
     */
    private fun aggregateScores(scores: List<PassiveScores>): PassiveScores {
        if (scores.isEmpty()) {
            return PassiveScores(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        }
        
        return PassiveScores(
            cnnScore = scores.map { it.cnnScore }.median(),
            lbpScore = scores.map { it.lbpScore }.median(),
            frequencyScore = scores.map { it.frequencyScore }.median(),
            colorScore = scores.map { it.colorScore }.median(),
            motionScore = scores.map { it.motionScore }.median(),
            textureScore = scores.map { it.textureScore }.median(),
            depthScore = scores.map { it.depthScore }.median(),
            finalScore = scores.map { it.finalScore }.median()
        )
    }
    
    /**
     * Extension function to calculate median of a list.
     */
    private fun List<Float>.median(): Float {
        if (isEmpty()) return 0f
        val sorted = this.sorted()
        return if (sorted.size % 2 == 0) {
            (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
        } else {
            sorted[sorted.size / 2]
        }
    }
    
    /**
     * Resets the detector for a new detection session.
     */
    fun reset() {
        frameScores.clear()
        motionAnalyzer.reset()
        depthAnalyzer.reset()
        blinkDetector.reset()
        currentStage = LivenessStage.Initializing
        Log.d(TAG, "Liveness detector reset")
    }
    
    /**
     * Releases all resources.
     */
    fun release() {
        cnnInference.close()
        Log.d(TAG, "Liveness detector released")
    }
}
