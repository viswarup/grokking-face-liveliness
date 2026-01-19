package com.uidai.livenessdetection.ml

import android.util.Log
import com.uidai.livenessdetection.domain.models.AttackType
import com.uidai.livenessdetection.domain.models.PassiveScores

/**
 * Fusion scorer that combines all detection method scores into a final decision.
 * Uses weighted combination based on each method's reliability.
 * 
 * CNN is now RE-ENABLED using ONNX Runtime.
 */
class FusionScorer {
    
    companion object {
        private const val TAG = "FusionScorer"
        
        // Weights for each detection component (sum = 1.0)
        // CNN is back! Using ONNX Runtime now.
        private const val WEIGHT_CNN = 0.40f        // Primary detection (ONNX MiniFASNet)
        private const val WEIGHT_LBP = 0.12f        // Texture analysis
        private const val WEIGHT_FREQUENCY = 0.15f  // Screen/moiré detection
        private const val WEIGHT_COLOR = 0.08f      // Skin tone
        private const val WEIGHT_MOTION = 0.10f     // Motion patterns
        private const val WEIGHT_TEXTURE = 0.05f    // Sharpness/blur
        private const val WEIGHT_DEPTH = 0.10f      // 3D validation
        
        // Decision thresholds
        const val PASS_THRESHOLD = 0.60f
        const val FAIL_THRESHOLD = 0.40f
    }
    
    /**
     * Calculates the final weighted score from all detection methods.
     */
    fun calculateFinalScore(
        cnnScore: Float,
        lbpScore: Float,
        frequencyScore: Float,
        colorScore: Float,
        motionScore: Float,
        textureScore: Float,
        depthScore: Float
    ): PassiveScores {
        
        val finalScore = (
            WEIGHT_CNN * cnnScore +
            WEIGHT_LBP * lbpScore +
            WEIGHT_FREQUENCY * frequencyScore +
            WEIGHT_COLOR * colorScore +
            WEIGHT_MOTION * motionScore +
            WEIGHT_TEXTURE * textureScore +
            WEIGHT_DEPTH * depthScore
        )
        
        Log.d(TAG, """
            === Score Breakdown (ONNX CNN) ===
            CNN:       ${String.format("%.3f", cnnScore)} × $WEIGHT_CNN = ${String.format("%.3f", cnnScore * WEIGHT_CNN)}
            LBP:       ${String.format("%.3f", lbpScore)} × $WEIGHT_LBP = ${String.format("%.3f", lbpScore * WEIGHT_LBP)}
            Frequency: ${String.format("%.3f", frequencyScore)} × $WEIGHT_FREQUENCY = ${String.format("%.3f", frequencyScore * WEIGHT_FREQUENCY)}
            Color:     ${String.format("%.3f", colorScore)} × $WEIGHT_COLOR = ${String.format("%.3f", colorScore * WEIGHT_COLOR)}
            Motion:    ${String.format("%.3f", motionScore)} × $WEIGHT_MOTION = ${String.format("%.3f", motionScore * WEIGHT_MOTION)}
            Texture:   ${String.format("%.3f", textureScore)} × $WEIGHT_TEXTURE = ${String.format("%.3f", textureScore * WEIGHT_TEXTURE)}
            Depth:     ${String.format("%.3f", depthScore)} × $WEIGHT_DEPTH = ${String.format("%.3f", depthScore * WEIGHT_DEPTH)}
            ====================================
            FINAL SCORE: ${String.format("%.3f", finalScore)} (Pass>$PASS_THRESHOLD, Fail<$FAIL_THRESHOLD)
        """.trimIndent())
        
        return PassiveScores(
            cnnScore = cnnScore,
            lbpScore = lbpScore,
            frequencyScore = frequencyScore,
            colorScore = colorScore,
            motionScore = motionScore,
            textureScore = textureScore,
            depthScore = depthScore,
            finalScore = finalScore
        )
    }
    
    /**
     * Gets the decision based on the final score.
     */
    fun getDecision(finalScore: Float): Decision {
        val decision = when {
            finalScore >= PASS_THRESHOLD -> Decision.PASS
            finalScore <= FAIL_THRESHOLD -> Decision.FAIL
            else -> Decision.UNCERTAIN
        }
        Log.d(TAG, "Decision: $decision (score: ${String.format("%.3f", finalScore)})")
        return decision
    }
    
    /**
     * Detects the type of attack based on individual scores.
     */
    fun detectAttackType(scores: PassiveScores): AttackType {
        return when {
            // CNN primary: if CNN says fake with high confidence
            scores.cnnScore < 0.3f -> 
                if (scores.frequencyScore < 0.5f) AttackType.SCREEN_REPLAY 
                else AttackType.PRINTED_PHOTO
            
            // Screen Replay: Low frequency score (moiré patterns)
            scores.frequencyScore < 0.4f -> 
                AttackType.SCREEN_REPLAY
            
            // Printed Photo: Low LBP (flat texture), low motion
            scores.lbpScore < 0.4f && scores.motionScore < 0.4f -> 
                AttackType.PRINTED_PHOTO
            
            // 2D Attack: Low depth variation (flat surface)
            scores.depthScore < 0.35f -> 
                AttackType.MASK_2D
            
            // Video Replay: Low motion variance
            scores.motionScore < 0.4f -> 
                AttackType.VIDEO_REPLAY
            
            // 3D Mask: Abnormal color and texture
            scores.colorScore < 0.45f && scores.textureScore < 0.45f -> 
                AttackType.MASK_3D
            
            else -> AttackType.UNKNOWN
        }.also {
            Log.d(TAG, "Detected attack type: $it")
        }
    }
    
    /**
     * Decision result from the fusion scorer.
     */
    enum class Decision {
        PASS,       // High confidence real face
        FAIL,       // High confidence spoof
        UNCERTAIN   // Needs active liveness
    }
}
