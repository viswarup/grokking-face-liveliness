package com.uidai.livenessdetection.ml

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Traditional Computer Vision analyzer for liveness detection.
 * Implements various texture and image analysis techniques using pure Android APIs.
 * No OpenCV dependency - uses Android's built-in Bitmap APIs.
 */
class TraditionalCVAnalyzer {
    
    companion object {
        private const val TAG = "TraditionalCVAnalyzer"
    }
    
    /**
     * Calculates Local Binary Pattern (LBP) texture score.
     * Detects texture uniformity - real faces have more natural texture variation.
     * @return Score between 0.0 (likely spoof) and 1.0 (likely real)
     */
    fun calculateLBPScore(faceBitmap: Bitmap): Float {
        return try {
            val grayPixels = bitmapToGrayscale(faceBitmap)
            val width = faceBitmap.width
            val height = faceBitmap.height
            
            // Calculate LBP histogram
            val histogram = IntArray(256)
            
            for (y in 1 until height - 1) {
                for (x in 1 until width - 1) {
                    val center = grayPixels[y * width + x]
                    var lbp = 0
                    
                    // 8 neighbors in clockwise order
                    if (grayPixels[(y - 1) * width + (x - 1)] >= center) lbp = lbp or 1
                    if (grayPixels[(y - 1) * width + x] >= center) lbp = lbp or 2
                    if (grayPixels[(y - 1) * width + (x + 1)] >= center) lbp = lbp or 4
                    if (grayPixels[y * width + (x + 1)] >= center) lbp = lbp or 8
                    if (grayPixels[(y + 1) * width + (x + 1)] >= center) lbp = lbp or 16
                    if (grayPixels[(y + 1) * width + x] >= center) lbp = lbp or 32
                    if (grayPixels[(y + 1) * width + (x - 1)] >= center) lbp = lbp or 64
                    if (grayPixels[y * width + (x - 1)] >= center) lbp = lbp or 128
                    
                    histogram[lbp]++
                }
            }
            
            // Calculate histogram uniformity (entropy-like measure)
            val totalPixels = (width - 2) * (height - 2)
            var entropy = 0.0
            for (count in histogram) {
                if (count > 0) {
                    val p = count.toDouble() / totalPixels
                    entropy -= p * kotlin.math.ln(p)
                }
            }
            
            // Normalize entropy to score (higher entropy = more texture = likely real)
            // Max entropy is ln(256) ≈ 5.54
            val normalizedEntropy = (entropy / 5.54).toFloat()
            
            // Real faces typically have entropy 0.6-0.9
            val score = when {
                normalizedEntropy < 0.3f -> 0.3f + normalizedEntropy  // Too uniform (print)
                normalizedEntropy > 0.95f -> 0.7f  // Too random (noise)
                else -> 0.5f + normalizedEntropy * 0.4f
            }
            
            Log.d(TAG, "LBP score: ${String.format("%.3f", score)} (entropy: ${String.format("%.3f", normalizedEntropy)})")
            score.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating LBP: ${e.message}")
            0.6f
        }
    }
    
    /**
     * Calculates frequency domain score.
     * Detects moiré patterns common in screen photos.
     * @return Score between 0.0 (likely spoof) and 1.0 (likely real)
     */
    fun calculateFrequencyScore(faceBitmap: Bitmap): Float {
        return try {
            val grayPixels = bitmapToGrayscale(faceBitmap)
            val width = faceBitmap.width
            val height = faceBitmap.height
            
            // Calculate horizontal and vertical gradients
            var highFreqEnergy = 0.0
            var lowFreqEnergy = 0.0
            
            for (y in 1 until height - 1) {
                for (x in 1 until width - 1) {
                    val dx = abs(grayPixels[y * width + (x + 1)] - grayPixels[y * width + (x - 1)])
                    val dy = abs(grayPixels[(y + 1) * width + x] - grayPixels[(y - 1) * width + x])
                    
                    val gradient = sqrt((dx * dx + dy * dy).toDouble())
                    
                    // High frequency = large, sudden changes
                    if (gradient > 50) {
                        highFreqEnergy += gradient
                    } else {
                        lowFreqEnergy += gradient
                    }
                }
            }
            
            val totalEnergy = highFreqEnergy + lowFreqEnergy
            val highFreqRatio = if (totalEnergy > 0) (highFreqEnergy / totalEnergy).toFloat() else 0.5f
            
            // Screens typically show high-frequency periodic patterns (moiré)
            // Real faces have balanced frequency content
            val score = when {
                highFreqRatio < 0.05f -> 0.4f  // Too smooth (blurry photo)
                highFreqRatio > 0.4f -> 0.5f   // Too much high freq (moiré/noise)
                else -> 0.6f + (1f - abs(highFreqRatio - 0.15f) / 0.15f) * 0.3f
            }
            
            Log.d(TAG, "Frequency score: ${String.format("%.3f", score)} (highFreqRatio: ${String.format("%.3f", highFreqRatio)})")
            score.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating frequency: ${e.message}")
            0.6f
        }
    }
    
    /**
     * Analyzes color distribution for natural skin tones.
     * Real faces have characteristic YCrCb color distribution.
     * @return Score between 0.0 (likely spoof) and 1.0 (likely real)
     */
    fun calculateColorScore(faceBitmap: Bitmap): Float {
        return try {
            val width = faceBitmap.width
            val height = faceBitmap.height
            val pixels = IntArray(width * height)
            faceBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            
            var skinPixels = 0
            var totalPixels = 0
            
            // Sample center region (face area)
            val startX = width / 4
            val endX = 3 * width / 4
            val startY = height / 4
            val endY = 3 * height / 4
            
            for (y in startY until endY) {
                for (x in startX until endX) {
                    val pixel = pixels[y * width + x]
                    val r = Color.red(pixel)
                    val g = Color.green(pixel)
                    val b = Color.blue(pixel)
                    
                    // Convert to YCrCb
                    val cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
                    val cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
                    
                    // Check if pixel is in typical skin color range
                    if (cr in 133.0..173.0 && cb in 77.0..127.0) {
                        skinPixels++
                    }
                    totalPixels++
                }
            }
            
            val skinRatio = skinPixels.toFloat() / totalPixels.toFloat()
            
            // Real faces typically have 40-75% skin pixels in the face region
            val score = when {
                skinRatio < 0.25f -> 0.4f  // Too few skin pixels (mask/unusual lighting)
                skinRatio > 0.85f -> 0.5f  // Suspiciously uniform
                skinRatio in 0.35f..0.75f -> 0.7f + (1f - abs(skinRatio - 0.55f) / 0.2f) * 0.25f
                else -> 0.55f
            }
            
            Log.d(TAG, "Color score: ${String.format("%.3f", score)} (skin ratio: ${String.format("%.3f", skinRatio)})")
            score.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating color: ${e.message}")
            0.6f
        }
    }
    
    /**
     * Calculates texture sharpness score using Laplacian variance.
     * Real faces have natural texture; prints/screens often have artifacts.
     * @return Score between 0.0 (likely spoof) and 1.0 (likely real)
     */
    fun calculateTextureScore(faceBitmap: Bitmap): Float {
        return try {
            val grayPixels = bitmapToGrayscale(faceBitmap)
            val width = faceBitmap.width
            val height = faceBitmap.height
            
            // Calculate Laplacian (second derivative for edge detection)
            var laplacianSum = 0.0
            var laplacianSqSum = 0.0
            var count = 0
            
            for (y in 1 until height - 1) {
                for (x in 1 until width - 1) {
                    // Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
                    val lap = (-4 * grayPixels[y * width + x] +
                            grayPixels[(y - 1) * width + x] +
                            grayPixels[(y + 1) * width + x] +
                            grayPixels[y * width + (x - 1)] +
                            grayPixels[y * width + (x + 1)]).toDouble()
                    laplacianSum += lap
                    laplacianSqSum += lap * lap
                    count++
                }
            }
            
            // Calculate variance of Laplacian
            val mean = laplacianSum / count
            val variance = (laplacianSqSum / count) - (mean * mean)
            val stdDev = sqrt(variance)
            
            // Higher variance = sharper image (usually good for real faces)
            // Normalize to reasonable range
            val normalizedVar = (stdDev / 50.0).coerceIn(0.0, 2.0).toFloat()
            
            val score = when {
                normalizedVar < 0.2f -> 0.4f   // Too blurry
                normalizedVar > 1.5f -> 0.6f   // Artificially sharp
                else -> 0.55f + normalizedVar * 0.35f
            }
            
            Log.d(TAG, "Texture score: ${String.format("%.3f", score)} (laplacian stdDev: ${String.format("%.2f", stdDev)})")
            score.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating texture: ${e.message}")
            0.6f
        }
    }
    
    /**
     * Converts a bitmap to grayscale pixel array.
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
            // Standard grayscale conversion
            grayPixels[i] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }
        
        return grayPixels
    }
}
