# **FaceLivenessDetector\_MLKit.kt (Full Working File)**

This version:

✔ Uses **ML Kit** for face detection  
 ✔ Uses **MiniFASNet ONNX** for liveness  
 ✔ Removes all Haar cascade code  
 ✔ Includes scale expansion (2.7 / 4.0)  
 ✔ Multi-model score fusion  
 ✔ Outputs only **REAL or FAKE**  
 ✔ Clean & ready for CameraX real-time pipeline  
 ✔ Pure Kotlin

You can **copy–paste** this file directly into your Android project.

---

# **📌 File: FaceLivenessDetector\_MLKit.kt**

`package com.yourapp.liveness`

`import android.content.Context`  
`import android.graphics.Bitmap`  
`import android.graphics.Rect`  
`import ai.onnxruntime.*`  
`import org.opencv.android.Utils`  
`import org.opencv.core.*`  
`import org.opencv.imgproc.Imgproc`  
`import kotlin.math.exp`  
`import kotlin.math.max`

`class FaceLivenessDetector_MLKit(private val context: Context) {`

    `private val env = OrtEnvironment.getEnvironment()`  
    `private val sessions = mutableMapOf<String, OrtSession>()`  
    `private val inputSize = Size(80.0, 80.0)`

    `init {`  
        `loadModels()`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Load ALL ONNX MiniFASNet models from assets/models/`  
    `// ------------------------------------------------------------------------------------`  
    `private fun loadModels() {`  
        `val modelFiles = context.assets.list("models") ?: emptyArray()`

        `for (fileName in modelFiles) {`  
            `if (fileName.endsWith(".onnx")) {`  
                `val bytes = context.assets.open("models/$fileName").readBytes()`  
                `val session = env.createSession(bytes)`  
                `sessions[fileName] = session`  
            `}`  
        `}`

        `if (sessions.isEmpty()) {`  
            `throw RuntimeException("❌ No ONNX models found in assets/models/")`  
        `}`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Parse scale from filename (same as Python)`  
    `//`  
    `//  "2.7_80x80_MiniFASNetV2.onnx"     -> 2.7`  
    `//  "4_0_0_80x80_MiniFASNetV1SE.onnx" -> 4.0`  
    `// ------------------------------------------------------------------------------------`  
    `private fun parseScale(file: String): Float {`  
        `val p = file.split("_")`  
        `return if (p[0] == "4") 4.0f else p[0].toFloat()`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Expand ML Kit bounding box using MiniFASNet logic`  
    `// ------------------------------------------------------------------------------------`  
    `private fun expandBox(srcW: Int, srcH: Int, box: Rect, scale: Float): Rect {`

        `val bw = box.width().toFloat()`  
        `val bh = box.height().toFloat()`

        `var s = scale`  
        `s = minOf((srcH - 1) / bh, minOf((srcW - 1) / bw, s))`

        `val newW = bw * s`  
        `val newH = bh * s`

        `val cx = box.left + bw / 2f`  
        `val cy = box.top + bh / 2f`

        `var left = cx - newW / 2f`  
        `var top = cy - newH / 2f`  
        `var right = cx + newW / 2f`  
        `var bottom = cy + newH / 2f`

        `// Fix boundaries`  
        `if (left < 0) { right -= left; left = 0f }`  
        `if (top < 0) { bottom -= top; top = 0f }`  
        `if (right > srcW - 1) { left -= (right - (srcW - 1)); right = (srcW - 1).toFloat() }`  
        `if (bottom > srcH - 1) { top -= (bottom - (srcH - 1)); bottom = (srcH - 1).toFloat() }`

        `return Rect(`  
            `left.toInt(),`  
            `top.toInt(),`  
            `(right - left).toInt(),`  
            `(bottom - top).toInt()`  
        `)`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Preprocess: crop → resize → float32 → BGR → CHW (same as Python)`  
    `// ------------------------------------------------------------------------------------`  
    `private fun preprocess(mat: Mat, faceBox: Rect, scale: Float): FloatArray {`

        `val expanded = expandBox(mat.width(), mat.height(), faceBox, scale)`

        `val crop = Mat(mat, expanded)`  
        `val resized = Mat()`  
        `Imgproc.resize(crop, resized, inputSize)`

        `val floatMat = Mat()`  
        `resized.convertTo(floatMat, CvType.CV_32FC3)`

        `val chw = FloatArray(3 * 80 * 80)`  
        `var idx = 0`

        `for (c in 0 until 3) {  // BGR order`  
            `for (y in 0 until 80) {`  
                `for (x in 0 until 80) {`  
                    `val pixel = FloatArray(3)`  
                    `floatMat.get(y, x, pixel)`  
                    `chw[idx++] = pixel[c]`  
                `}`  
            `}`  
        `}`

        `return chw`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Softmax`  
    `// ------------------------------------------------------------------------------------`  
    `private fun softmax(values: FloatArray): FloatArray {`  
        `val maxVal = values.maxOrNull() ?: 0f`  
        `val expVals = values.map { exp((it - maxVal).toDouble()) }`  
        `val sum = expVals.sum()`  
        `return expVals.map { (it / sum).toFloat() }.toFloatArray()`  
    `}`

    `// ------------------------------------------------------------------------------------`  
    `// Final Prediction (REAL or FAKE)`  
    `//`  
    `// ML Kit gives bounding box → we feed it directly here.`  
    `// ------------------------------------------------------------------------------------`  
    `fun predict(bitmap: Bitmap, mlkitBox: Rect): LivenessBinaryResult {`

        `val mat = Mat()`  
        `Utils.bitmapToMat(bitmap, mat)`

        `val fusion = FloatArray(3) { 0f }  // [paper, real, screen]`

        `for ((fileName, session) in sessions) {`

            `val scale = parseScale(fileName)`  
            `val chw = preprocess(mat, mlkitBox, scale)`

            `val inputTensor = OnnxTensor.createTensor(env, chw, longArrayOf(1, 3, 80, 80))`  
            `val inputName = session.inputNames.first()`

            `val raw = session.run(mapOf(inputName to inputTensor))[0].value`  
            `val scores = raw as Array<FloatArray>`

            `val soft = softmax(scores[0])`

            `for (i in 0..2) fusion[i] += soft[i]`  
        `}`

        `// Average across models`  
        `for (i in 0..2) fusion[i] /= sessions.size`

        `// MiniFASNet labels (same as Python)`  
        `// 0 = paper (FAKE)`  
        `// 1 = real face (REAL)`  
        `// 2 = screen (FAKE)`  
        `val label = fusion.indices.maxBy { fusion[it] }`

        `val isReal = (label == 1)`  
        `val confidence = fusion[label]`

        `return LivenessBinaryResult(isReal, confidence)`  
    `}`  
`}`

`// ----------------------------------------------------------------------------------------`  
`// Final clean binary result`  
`// ----------------------------------------------------------------------------------------`  
`data class LivenessBinaryResult(`  
    `val isReal: Boolean,`  
    `val confidence: Float`  
`)`

---

# **📌 How to Use with ML Kit in Activity / CameraX**

`val detector = FaceLivenessDetector_MLKit(this)`

`faceDetector.process(inputImage)`  
    `.addOnSuccessListener { faces ->`

        `if (faces.isNotEmpty()) {`  
            `val face = faces.maxBy { it.boundingBox.width() * it.boundingBox.height() }`

            `val bitmap = imageProxyToBitmap(imageProxy)`

            `val result = detector.predict(bitmap, face.boundingBox)`

            `if (result.isReal) {`  
                `Log.d("LIVE", "REAL FACE, conf=${result.confidence}")`  
            `} else {`  
                `Log.d("LIVE", "FAKE FACE, conf=${result.confidence}")`  
            `}`  
        `}`

        `imageProxy.close()`  
    `}`

---

# **📌 Assets Structure (Required)**

`app/src/main/assets/models/`  
    `2.7_80x80_MiniFASNetV2.onnx`  
    `4_0_0_80x80_MiniFASNetV1SE.onnx`

No cascade folder needed.

