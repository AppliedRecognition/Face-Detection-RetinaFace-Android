package com.appliedrec.verid3.facedetection.testapp

import android.annotation.SuppressLint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.appliedrec.verid3.facedetection.testapp.ui.theme.FaceDetectionTheme
import java.io.File


class MainActivity : ComponentActivity() {

    val faceDetectionViewModel: FaceDetectionViewModel by viewModels()

    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FaceDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    val context = LocalContext.current
                    val isModelLoaded by faceDetectionViewModel.isLoaded.collectAsState()
                    val annotatedBitmap by faceDetectionViewModel.annotatedBitmap.collectAsState()
                    val modelPath by faceDetectionViewModel.detectionModelPath.collectAsState()
                    val useNnapi by faceDetectionViewModel.detectionUseNnapi.collectAsState()
                    val nnapiFlags by faceDetectionViewModel.detectionNnapiFlags.collectAsState()
                    val minDetectionSpeedMs by faceDetectionViewModel.minDetectionSpeedMs.collectAsState()
                    val maxDetectionSpeedMs by faceDetectionViewModel.maxDetectionSpeedMs.collectAsState()
                    val medianDetectionSpeedMs by faceDetectionViewModel.medianDetectionSpeedMs.collectAsState()
                    val scrollState = rememberScrollState()
                    if (isModelLoaded) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .verticalScroll(scrollState)
                                .padding(padding)
                                .padding(horizontal = 16.dp)
                        ) {
                            PhotoPicker(context) { bitmap ->
                                faceDetectionViewModel.setBitmap(bitmap)
                            }
                            modelPath?.let { path ->
                                Row {
                                    Text("Model path")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text(File(path).name)
                                }
                            }
                            useNnapi?.let { nnapi ->
                                Row {
                                    Text("Use NNAPI")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text(if (nnapi) "Yes" else "No")
                                }

                            }
                            nnapiFlags?.let { flags ->
                                Row {
                                    Text("NNAPI flags")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text(flags.toString())
                                }
                            }
                            minDetectionSpeedMs?.let { minSpeed ->
                                Row {
                                    Text("Min. detection speed")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text("%d ms".format(minSpeed))
                                }
                            }
                            maxDetectionSpeedMs?.let { maxSpeed ->
                                Row {
                                    Text("Max detection speed")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text("%d ms".format(maxSpeed))
                                }
                            }
                            medianDetectionSpeedMs?.let { medSpeed ->
                                Row {
                                    Text("Median detection speed")
                                    Spacer(
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text("%d ms".format(medSpeed))
                                }
                            }
                            annotatedBitmap?.let { bitmap ->
                                Image(
                                    bitmap = bitmap.asImageBitmap(),
                                    contentDescription = "Selected image",
                                    Modifier
                                        .fillMaxWidth()
                                        .aspectRatio(bitmap.width.toFloat() / bitmap.height.toFloat()),
                                    contentScale = ContentScale.Fit
                                )
                            }
                        }
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center,
                        ) {
                            CircularProgressIndicator()
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    FaceDetectionTheme {
        Greeting("Android")
    }
}