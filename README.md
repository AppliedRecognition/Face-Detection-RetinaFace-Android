# RetinaFace face detection for Android

Android face detection library using RetinaFace model compatible with Ver-ID SDK.

The [FaceDetectionRetinaFace](./lib/src/main/java/com/appliedrec/verid3/facedetection/retinaface/FaceDetectionRetinaFace.kt) class implements the [FaceDetection](https://github.com/AppliedRecognition/Ver-ID-Common-Types-Android/blob/main/lib/src/main/java/com/appliedrec/verid3/common/FaceDetection.kt) interface from the [Ver-ID common types](https://github.com/AppliedRecognition/Ver-ID-Common-Types-Android) library, making it simple to use with Ver-ID SDK components.

## Installation

1. Add your GitHub user name and access token to your **gradle.properties** file (either one in your project or global one):

    ```groovy
    gpr.user=YourUserName
    gpr.token=ghp_******
    ```
2. Add Ver-ID repository in your build file:

    ```kotlin
    dependencyResolutionManagement {
        repositories {
            maven {
                url = uri("https://maven.pkg.github.com/AppliedRecognition/Ver-ID-3D-Android-Libraries")
                credentials {
                    username = settings.extra["gpr.user"] as String?
                    password = settings.extra["gpr.token"] as String?
                }
            }
        }
    }
    ```
3. Add the dependency in your **build.gradle.kts** file:

    ```kotlin
    dependencies {
        implementation("com.appliedrec:verid3-face-detection-retinaface:1.0.0")
        implementation("com.appliedrec:verid3-serialization:1.0.1")
    }
    ```
4. Sync and build your project.

## Usage

The face detector is primarily meant to rapidly capture one face for face recognition but the API allows for up to 100 faces to be requested. You can set the limit in the `limit` parameter.

```kotlin
class MyActivity : ComponentActivity() {
    
    // Detect face in a bitmap and draw the image with the face in an image view
    
    fun detectFaceInImage(bitmap: Bitmap, imageView: ImageView) {

        // Launch in a lifecycle scope
        lifecycleScope.launch {
        
            // Create face detection instance
            val faceDetection = FaceDetectionRetinaFace.create(applicationContext)
            
            // Convert bitmap to Ver-ID image
            val image = Image.fromBitmap(bitmap)
            
            // Detect one face
            faceDetection.detectFacesInImage(image, 1).firstOrNull()?.let { face ->
            
                // If face detected create a copy of the input bitmap
                val annotated = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                
                // Wrap the bitmap in a canvas
                val canvas = Canvas(annotated)
                
                // Define paint for the face outline
                val paint = Paint().apply {
                    color = Color.GREEN
                    style = Paint.Style.STROKE
                    strokeWidth = bitmap.width.toFloat() / 100f
                    isAntiAlias = true
                }
                
                // Draw the face boundary on the canvas
                canvas.drawRect(face.bounds, paint)
                
                // Switch to UI thread
                withContext(Dispatchers.Main) {
                
                    // Draw the bitmap in the image view
                    imageView.setImageBitmap(annotated)
                }
            } ?: run {
                // No face detected, draw the original bitmap in the image view
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(bitmap)
                }
            }
        }
    }
}
```

Note that the `FaceDetectionRetinaFace.create` function runs a calibration to determine which variant of the face detection model performs best on the device. This calibration can take a few seconds to run. The result of the calibration is stored in shared preferences and the calibration is not rerun unless the `FaceDetectionRetinaFace.create`'s parameter `forceCalibrate` is set to `true`.

If, for some reason, you want to avoid the calibration you can call the `FaceDetectionRetinaFace` class constructor directly with the model file variant and NNAPI options. For example, to run inference on non-quantised model without using NNAPI, you can construct the face detection instance like this:

```kotlin
FaceDetectionRetinaFace(context, SessionConfiguration.FP32)
```
However, be careful. Not all configurations are supported on all devices and the constructor may throw an exception.

