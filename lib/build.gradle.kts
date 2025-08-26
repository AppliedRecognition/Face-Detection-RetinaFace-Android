import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.dokka)
    alias(libs.plugins.vanniktech.publish)
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
    ndkVersion = "28.2.13676358"
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    api(libs.verid.common)
    implementation(libs.kotlinx.coroutines)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.verid.common.serialization)
}

mavenPublishing {
    coordinates("com.appliedrec", "face-detection-retinaface")
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
        licenses {
            license {
                name.set("Commercial")
                url.set("https://raw.githubusercontent.com/AppliedRecognition/Face-Detection-RetinaFace-Android/refs/heads/main/LICENCE.txt")
            }
        }
        scm {
            connection.set("scm:git:git://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android.git")
            developerConnection.set("scm:git:ssh://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android.git")
            url.set("https://github.com/AppliedRecognition/Face-Detection-RetinaFace-Android")
        }
    }
    publishToMavenCentral(automaticRelease = true)
}

signing {
    useGpgCmd()
    sign(publishing.publications)
}

tasks.dokkaHtml {
    outputDirectory.set(rootDir.resolve("docs"))
}