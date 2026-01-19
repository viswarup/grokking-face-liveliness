package com.uidai.livenessdetection.domain.models

import android.graphics.PointF
import android.graphics.Rect

/**
 * Data class representing extracted face information from a camera frame.
 */
data class FaceData(
    /** Bounding box of the detected face */
    val boundingBox: Rect,
    
    /** List of facial landmarks (468 points from MediaPipe Face Mesh) */
    val landmarks: List<PointF>,
    
    /** Head pose Euler angles [yaw, pitch, roll] in degrees */
    val eulerAngles: FloatArray,
    
    /** Current Eye Aspect Ratio (used for blink detection) */
    val eyeAspectRatio: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FaceData

        if (boundingBox != other.boundingBox) return false
        if (landmarks != other.landmarks) return false
        if (!eulerAngles.contentEquals(other.eulerAngles)) return false
        if (eyeAspectRatio != other.eyeAspectRatio) return false

        return true
    }

    override fun hashCode(): Int {
        var result = boundingBox.hashCode()
        result = 31 * result + landmarks.hashCode()
        result = 31 * result + eulerAngles.contentHashCode()
        result = 31 * result + eyeAspectRatio.hashCode()
        return result
    }
}
