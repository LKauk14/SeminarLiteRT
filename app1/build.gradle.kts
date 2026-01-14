plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.seminarlitert"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.example.seminarlitert"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"


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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        mlModelBinding = true
    }
}



dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
   
    // Google Play Services LiteRT
    implementation("com.google.android.gms:play-services-tflite-java:16.1.0")
    implementation("com.google.android.gms:play-services-tflite-support:16.1.0")
// optional GPU Delegate
    implementation("com.google.android.gms:play-services-tflite-gpu:16.1.0")
}