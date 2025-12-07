package com.example.seminarlitert;

import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.android.gms.tasks.Task;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

public class MainActivity extends AppCompatActivity {

    //Modellabhängig
    private static final int IMAGE_PICK_CODE = 1001;
    private static final int IMAGE_SIZE = 224;
    private InterpreterApi interpreter;
    private Classifier classifier;

    private Button buttonSelect;
    private Button buttonUpload;
    private ImageView imageView;
    private TextView textResult;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        imageView = findViewById(R.id.imageView3);
        textResult = findViewById(R.id.textView);
        buttonSelect = findViewById(R.id.button);
        buttonUpload = findViewById(R.id.button);

        classifier = new Classifier(this, "model.tflite", IMAGE_SIZE);
        buttonSelect.setOnClickListener(v -> pickImageFromGallery());
    }

    private void pickImageFromGallery() {
    }
}