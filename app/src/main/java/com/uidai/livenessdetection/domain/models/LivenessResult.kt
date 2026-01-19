package com.uidai.livenessdetection.domain.models

/**
 * Represents the different stages of the liveness detection process.
 */
sealed class LivenessStage {
    /** Initial state when the system is being set up */
    object Initializing : LivenessStage()
    
    /** Waiting for face to be properly aligned in the frame */
    object FaceAlignment : LivenessStage()
    
    /** Passive liveness analysis in progress */
    data class PassiveAnalysis(
        val frameCount: Int,
        val totalFrames: Int,
        val currentScores: PassiveScores? = null  // Added: current frame scores for UI display
    ) : LivenessStage()
    
    /** Active liveness prompt (blink detection) */
    data class ActivePrompt(
        val blinkCount: Int,
        val requiredBlinks: Int
    ) : LivenessStage()
    
    /** Detection completed with result */
    data class Completed(val result: LivenessResult) : LivenessStage()
}

/**
 * Final result of the liveness detection.
 */
data class LivenessResult(
    val isPassed: Boolean,
    val confidence: Float,
    val passiveScore: PassiveScores,
    val activeResult: ActiveResult?,
    val attackType: AttackType?,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Scores from all passive detection methods.
 */
data class PassiveScores(
    val cnnScore: Float,
    val lbpScore: Float,
    val frequencyScore: Float,
    val colorScore: Float,
    val motionScore: Float,
    val textureScore: Float,
    val depthScore: Float,
    val finalScore: Float
)

/**
 * Result from active liveness detection (blink detection).
 */
data class ActiveResult(
    val blinkDetected: Boolean,
    val blinkCount: Int,
    val blinkTimings: List<Long>,
    val isNatural: Boolean
)

/**
 * Types of spoofing attacks that can be detected.
 */
enum class AttackType {
    PRINTED_PHOTO,
    SCREEN_REPLAY,
    MASK_2D,
    MASK_3D,
    VIDEO_REPLAY,
    DEEPFAKE,
    UNKNOWN
}
