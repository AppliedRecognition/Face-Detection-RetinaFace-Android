import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.io.File

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.dokka)
    `maven-publish`
    signing
}

version = "1.0.0"

android {
    namespace = "com.appliedrec.verid3.facedetection.retinaface"
    compileSdk = 36

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    publishing {
        singleVariant("release") {}
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "27.1.12297006"
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.verid.common)
    implementation(libs.kotlinx.coroutines)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.verid.common.serialization)
}



val zipalignAar = tasks.register<Exec>("zipalignAar") {
    val inputAar = layout.buildDirectory.file("outputs/aar/${project.name}-release.aar").get().asFile
    val outputAar = inputAar

    group = "build"
    description = "Align AAR to 16KB"

    // Assume environment has ANDROID_SDK_ROOT or ANDROID_HOME set
    val sdkDir = System.getenv("ANDROID_SDK_ROOT")
        ?: System.getenv("ANDROID_HOME")
        ?: throw GradleException("ANDROID_SDK_ROOT or ANDROID_HOME environment variable not set.")

    val buildToolsVersion = android.buildToolsVersion
    val zipalignPath = File(sdkDir, "build-tools/$buildToolsVersion/zipalign").absolutePath

    doFirst {
        if (!File(zipalignPath).exists()) {
            throw GradleException("zipalign not found at $zipalignPath. Please ensure Build Tools $buildToolsVersion is installed.")
        }
        if (!inputAar.exists()) {
            throw GradleException("Release AAR not found: ${inputAar.absolutePath}. Run 'assembleRelease' first.")
        }
    }

    commandLine(
        zipalignPath,
        "-f", "-v", "16384",
        inputAar.absolutePath,
        outputAar.absolutePath
    )
}

publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "com.appliedrec"
            artifactId = "verid3-face-detection-retinaface"
            afterEvaluate {
                from(components["release"])
            }

            pom {
                name.set("RetinaFace Face Detection")
                description.set("RetinaFace face detection for Ver-ID SDK")
                url.set("https://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android")
                developers {
                    developer {
                        id.set("appliedrec")
                        name.set("Applied Recognition")
                        email.set("support@appliedrecognition.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android.git")
                    developerConnection.set("scm:git:ssh://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android.git")
                    url.set("https://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android")
                }
            }
        }
    }

    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/AppliedRecognition/Ver-ID-Releases-Android")
            credentials {
                username = project.findProperty("gpr.user") as String?
                password = project.findProperty("gpr.token") as String?
            }
        }
    }
}

signing {
    useGpgCmd()
    sign(publishing.publications["release"])
}

tasks.dokkaHtml {
    outputDirectory.set(rootDir.resolve("docs"))
}

tasks.named("assemble").configure {
    finalizedBy(zipalignAar)
}