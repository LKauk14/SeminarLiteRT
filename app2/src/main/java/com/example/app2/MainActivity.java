package com.example.app2;
import android.os.Bundle;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;



import org.tensorflow.lite.InterpreterApi;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "TFLitePlayServices";

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1️⃣ Initialize TFLite asynchronously
        TfLite.initialize(this)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void unused) {
                        Log.d(TAG, "Play Services TFLite initialized!");
                        try {
                            runModel();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(Exception e) {
                        Log.e(TAG, "TFLite initialization failed", e);
                    }
                });
    }

    private void runModel() throws IOException {
        // 2️⃣ Load model from assets
        MappedByteBuffer modelBuffer = InterpreterApi.loadModelFromAsset(this, "model.tflite");

        // 3️⃣ Create Interpreter
        InterpreterApi interpreter = InterpreterApi.create(modelBuffer);

        // 4️⃣ Prepare input & output (example: single float input, single float output)
        float[] input = new float[]{1.0f};
        float[] output = new float[1];

        // 5️⃣ Run inference
        interpreter.run(input, output);

        Log.d(TAG, "Model output: " + output[0]);
    }
}
