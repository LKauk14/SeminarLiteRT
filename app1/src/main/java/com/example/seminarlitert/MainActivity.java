package com.example.seminarlitert;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;


import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;


import com.google.android.gms.tasks.Task;
import com.google.android.gms.tflite.java.TfLite;

import org.tensorflow.lite.InterpreterApi;


import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    //Modellabhängig
    enum ClassifyMode {
        NORMAL,
        TENSOR_IMAGE
    }

    private static final int IMAGE_PICK_CODE = 1001;
    private static final int IMAGE_SIZE = 224;
    private InterpreterApi interpreter;
    private Classifier classifier;
    ClassifyMode mode = ClassifyMode.NORMAL;
    private Button buttonClassify;
    private Button buttonUpload;
    private ImageView imageView;
    private TextView textViewResult;
    private Bitmap selectedBitmap;

    private Switch switchMethod;

    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    loadImage(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        setContentView(R.layout.activity_main);

        switchMethod = findViewById(R.id.switchMethod);
        imageView = findViewById(R.id.imageView);
        buttonUpload = findViewById(R.id.buttonUpload);
        buttonClassify = findViewById(R.id.buttonClassify);
        textViewResult = findViewById(R.id.textViewResult);

        Task<Void> initializeTask = TfLite.initialize(this);
        initializeTask.addOnSuccessListener(a -> {
                    classifier = new Classifier(this, "mobilenetv1.tflite", "labels.txt", IMAGE_SIZE);
                })
                .addOnFailureListener(e -> {
                    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
                            e.getMessage()));
                });

        buttonUpload.setOnClickListener(v -> pickImageFromGallery());
        buttonClassify.setOnClickListener(v -> {

            if (selectedBitmap == null) {
                textViewResult.setText("Bitte zuerst ein Bild auswählen!");
                return;
            }
            mode = switchMethod.isChecked() ? ClassifyMode.TENSOR_IMAGE : ClassifyMode.NORMAL;

            switch (mode) {
                case NORMAL:
                    classifier.classify(selectedBitmap, result -> runOnUiThread(() -> {
                        textViewResult.setText("Ergebnis: " + result);
                    }));
                    break;
                case TENSOR_IMAGE:
                    classifier.classifyWithTensorImage(selectedBitmap, result -> runOnUiThread(() -> {
                        textViewResult.setText("Ergebnis: " + result);
                    }));
                    break;

            }
        });
    }


    private void pickImageFromGallery() {
        galleryLauncher.launch("image/*");
    }

    private void loadImage(Uri uri) {
        try {
            selectedBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            imageView.setImageBitmap(selectedBitmap);

        } catch (IOException e) {
            e.printStackTrace();
            textViewResult.setText("Fehler beim Laden des Bildes!");
        }
    }
}