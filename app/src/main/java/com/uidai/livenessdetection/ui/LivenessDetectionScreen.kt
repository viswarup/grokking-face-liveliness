package com.uidai.livenessdetection.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.uidai.livenessdetection.domain.models.*

/**
 * Main screen for liveness detection.
 * Displays camera preview with overlay and status panel.
 */
@Composable
fun LivenessDetectionScreen(
    viewModel: LivenessViewModel = viewModel()
) {
    val stage by viewModel.stage.collectAsState()
    val faceBox by viewModel.faceBox.collectAsState()
    val imageDimensions by viewModel.imageDimensions.collectAsState()
    val instructionText by viewModel.instructionText.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize()) {
        // Top Section: Camera Preview (60%)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.6f)
                .background(Color.Black)
        ) {
            // Camera Preview
            CameraPreview(
                onFrameAnalyzed = { bitmap, landmarks, eulerAngles, boundingBox, imageWidth, imageHeight ->
                    viewModel.updateFaceBox(boundingBox)
                    viewModel.processFrame(bitmap, landmarks, eulerAngles, imageWidth, imageHeight)
                },
                onNoFaceDetected = {
                    viewModel.updateFaceBox(null)
                }
            )
            
            // Face Bounding Box Overlay
            Canvas(modifier = Modifier.fillMaxSize()) {
                faceBox?.let { box ->
                    val color = when (stage) {
                        is LivenessStage.Completed -> {
                            if ((stage as LivenessStage.Completed).result.isPassed) 
                                Color.Green else Color.Red
                        }
                        is LivenessStage.PassiveAnalysis -> Color.Yellow
                        is LivenessStage.ActivePrompt -> Color(0xFFFFA000)
                        else -> Color.White
                    }
                    
                    // Get actual image dimensions
                    val (imageWidth, imageHeight) = imageDimensions
                    
                    // Calculate scale based on canvas size and image dimensions
                    // Note: Front camera is mirrored horizontally
                    val scaleX = size.width / imageWidth.toFloat()
                    val scaleY = size.height / imageHeight.toFloat()
                    
                    // Mirror the X coordinate for front camera
                    val mirroredLeft = imageWidth - box.right
                    
                    drawRect(
                        color = color,
                        topLeft = Offset(mirroredLeft * scaleX, box.top * scaleY),
                        size = Size(box.width() * scaleX, box.height() * scaleY),
                        style = Stroke(width = 4f)
                    )
                }
            }
            
            // Instruction Overlay at bottom of camera
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
                    .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 16.dp, vertical = 12.dp)
            ) {
                Text(
                    text = instructionText,
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center
                )
            }
        }
        
        // Bottom Section: Information Panel (40%)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight()
                .background(Color(0xFFF5F5F5))
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            // Status Panel
            StatusPanel(stage = stage)
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Stage-specific panels
            when (stage) {
                is LivenessStage.PassiveAnalysis -> {
                    PassiveLivenessPanel(stage as LivenessStage.PassiveAnalysis)
                }
                is LivenessStage.ActivePrompt -> {
                    ActiveLivenessPanel(stage as LivenessStage.ActivePrompt)
                }
                is LivenessStage.Completed -> {
                    FinalResultPanel(
                        result = (stage as LivenessStage.Completed).result,
                        onRetry = { viewModel.reset() }
                    )
                }
                else -> {
                    InitializingPanel()
                }
            }
        }
    }
}

/**
 * Panel showing current status and stage.
 */
@Composable
fun StatusPanel(stage: LivenessStage) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Live Status",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1565C0)
            )
            Spacer(modifier = Modifier.height(12.dp))
            
            val (stageText, statusColor) = when (stage) {
                is LivenessStage.Initializing -> "Initializing" to Color.Gray
                is LivenessStage.FaceAlignment -> "Face Alignment" to Color(0xFF2196F3)
                is LivenessStage.PassiveAnalysis -> "Passive Liveness Check" to Color(0xFFFFA000)
                is LivenessStage.ActivePrompt -> "Active Liveness Check" to Color(0xFFFF6F00)
                is LivenessStage.Completed -> {
                    if (stage.result.isPassed) "PASSED" to Color(0xFF4CAF50)
                    else "FAILED" to Color(0xFFF44336)
                }
            }
            
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(text = "Stage: ", fontWeight = FontWeight.Bold)
                Text(text = stageText)
                Spacer(modifier = Modifier.weight(1f))
                Box(
                    modifier = Modifier
                        .size(16.dp)
                        .background(statusColor, RoundedCornerShape(8.dp))
                )
            }
        }
    }
}

/**
 * Panel shown during initialization.
 */
@Composable
fun InitializingPanel() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Simple loading text instead of CircularProgressIndicator
            Text(
                text = "⏳",
                fontSize = 48.sp
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Preparing camera...",
                fontSize = 16.sp,
                color = Color.Gray
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Position your face in the frame",
                fontSize = 14.sp,
                color = Color.Gray,
                textAlign = TextAlign.Center
            )
        }
    }
}

/**
 * Panel showing passive analysis progress with live scores.
 */
@Composable
fun PassiveLivenessPanel(stage: LivenessStage.PassiveAnalysis) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "📊 Passive Analysis",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))
            
            LinearProgressIndicator(
                progress = stage.frameCount.toFloat() / stage.totalFrames.toFloat(),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp),
                color = Color(0xFF1565C0),
                trackColor = Color(0xFFE3F2FD)
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = "Frame ${stage.frameCount}/${stage.totalFrames}",
                fontSize = 14.sp,
                color = Color.Gray
            )
            
            // Show live scores if available
            stage.currentScores?.let { scores ->
                Spacer(modifier = Modifier.height(12.dp))
                
                // Score grid
                Column {
                    ScoreRow("🧠 CNN", scores.cnnScore, Color(0xFF1565C0))
                    ScoreRow("📐 LBP", scores.lbpScore, Color(0xFF7B1FA2))
                    ScoreRow("📡 Frequency", scores.frequencyScore, Color(0xFF00796B))
                    ScoreRow("🎨 Color", scores.colorScore, Color(0xFFE64A19))
                    ScoreRow("🌊 Motion", scores.motionScore, Color(0xFF0288D1))
                    ScoreRow("🔲 Texture", scores.textureScore, Color(0xFF5D4037))
                    ScoreRow("📏 Depth", scores.depthScore, Color(0xFF455A64))
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    Divider()
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    // Final score with color coding
                    val finalColor = when {
                        scores.finalScore >= 0.65f -> Color(0xFF4CAF50)  // Green - PASS
                        scores.finalScore <= 0.40f -> Color(0xFFF44336)  // Red - FAIL
                        else -> Color(0xFFFFA000)  // Orange - UNCERTAIN
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "⭐ FINAL SCORE",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = String.format("%.1f%%", scores.finalScore * 100),
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = finalColor
                        )
                    }
                }
            }
        }
    }
}

/**
 * Single score row with label, progress bar, and value.
 */
@Composable
fun ScoreRow(label: String, score: Float, color: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            modifier = Modifier.width(100.dp)
        )
        LinearProgressIndicator(
            progress = score,
            modifier = Modifier
                .weight(1f)
                .height(6.dp),
            color = color,
            trackColor = Color(0xFFE0E0E0)
        )
        Text(
            text = String.format("%.0f%%", score * 100),
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.width(40.dp),
            textAlign = TextAlign.End
        )
    }
}

/**
 * Panel for active liveness (blink detection).
 */
@Composable
fun ActiveLivenessPanel(stage: LivenessStage.ActivePrompt) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0)),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "👁️ Active Liveness Required",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFFE65100)
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "Please blink twice naturally",
                fontSize = 16.sp,
                color = Color(0xFFE65100)
            )
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Blink counter
            Row(
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier.fillMaxWidth()
            ) {
                repeat(stage.requiredBlinks) { index ->
                    val isCompleted = index < stage.blinkCount
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .background(
                                if (isCompleted) Color(0xFF4CAF50) else Color(0xFFE0E0E0),
                                RoundedCornerShape(24.dp)
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (isCompleted) "✓" else "${index + 1}",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            color = if (isCompleted) Color.White else Color.Gray
                        )
                    }
                    if (index < stage.requiredBlinks - 1) {
                        Spacer(modifier = Modifier.width(16.dp))
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "Blinks: ${stage.blinkCount} / ${stage.requiredBlinks}",
                fontSize = 14.sp,
                color = Color.Gray
            )
        }
    }
}

/**
 * Panel showing final result.
 */
@Composable
fun FinalResultPanel(result: LivenessResult, onRetry: () -> Unit) {
    val backgroundColor = if (result.isPassed) Color(0xFF4CAF50) else Color(0xFFF44336)
    val buttonColor = if (result.isPassed) Color(0xFF4CAF50) else Color(0xFFF44336)
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = backgroundColor),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = if (result.isPassed) "✓ LIVE FACE VERIFIED" else "✗ SPOOF DETECTED",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Text(
                text = if (result.isPassed) 
                    "No spoofing detected\nAuthentication successful" 
                    else "Authentication failed",
                fontSize = 16.sp,
                color = Color.White.copy(alpha = 0.9f),
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Confidence score
            Text(
                text = "Confidence: ${String.format("%.1f", result.confidence * 100)}%",
                fontSize = 14.sp,
                color = Color.White.copy(alpha = 0.8f)
            )
            
            // Attack type if failed
            if (!result.isPassed && result.attackType != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Detected: ${result.attackType.name.replace('_', ' ')}",
                    fontSize = 14.sp,
                    color = Color.White.copy(alpha = 0.8f)
                )
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Retry/Done button
            Button(
                onClick = onRetry,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = buttonColor
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = if (result.isPassed) "DONE" else "RETRY",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
