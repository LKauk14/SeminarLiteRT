package com.example.seminarlitert;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;


import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tflite.java.TfLite;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

import org.tensorflow.lite.InterpreterApi;


import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    //Modellabhängig
    private static final int IMAGE_PICK_CODE = 1001;
    private static final int IMAGE_SIZE = 224;
    private InterpreterApi interpreter;
    private Classifier classifier;

    private Button buttonClassify;
    private Button buttonUpload;
    private ImageView imageView;
    private TextView textViewResult;
    private Bitmap selectedBitmap;

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


        imageView = findViewById(R.id.imageView);
        buttonUpload = findViewById(R.id.buttonUpload);
        buttonClassify = findViewById(R.id.buttonClassify);
        textViewResult = findViewById(R.id.textViewResult);

        Task<Void> initializeTask = TfLite.initialize(this);
        initializeTask.addOnSuccessListener(a -> {
                    classifier = new Classifier(this, "mobilenetv1.tflite","labels.txt", IMAGE_SIZE);
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

            // Klassifizierung starten
            classifier.classify(selectedBitmap, result -> runOnUiThread(() -> {
                textViewResult.setText("Ergebnis: " + result);
            }));
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