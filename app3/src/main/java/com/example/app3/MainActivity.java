package com.example.app3;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.example.app3.R;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.LiteRtException;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_SIZE = 224;

    private Button buttonUpload;
    private Button buttonClassify;
    private ImageView imageView;
    private TextView textViewResult;
    private Bitmap selectedBitmap;
    private Switch switchAccelaratorMethod;
    private Classifier classifier;


    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    loadImage(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Layout setzen
        setContentView(R.layout.activity_main);

        // UI Elemente
        buttonUpload = findViewById(R.id.buttonUpload);
        buttonClassify = findViewById(R.id.buttonClassify);
        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        switchAccelaratorMethod = findViewById(R.id.switchAccelarator);
        switchAccelaratorMethod.setOnCheckedChangeListener(this::onSwitchChanged);


        // Classifier initialisieren
        try {
            classifier = new Classifier(this, "mobilenetv2.tflite", "labels1.txt", IMAGE_SIZE);
        } catch (LiteRtException e) {
            throw new RuntimeException(e);
        }


        // Button: Bild auswählen
        buttonUpload.setOnClickListener(v -> galleryLauncher.launch("image/*"));

        // Button: Klassifizieren
        buttonClassify.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                textViewResult.setText("Bitte zuerst ein Bild auswählen!");
                return;
            }

            try {
                classifier.classify(selectedBitmap, result -> runOnUiThread(() ->
                        textViewResult.setText("Ergebnis: " + result)
                ));
            } catch (LiteRtException e) {
                textViewResult.setText("Fehler bei Inferenz: " + e.getMessage());
            }
        });
    }

    private void onSwitchChanged(CompoundButton buttonView, boolean isChecked) {

        Accelerator newAccelerator;
        if (isChecked) {
            if (!classifier.isGpuSupported(this, "mobilenetv2.tflite")) {

                Toast.makeText(this,
                        "GPU nicht verfügbar",
                        Toast.LENGTH_SHORT
                ).show();
                buttonView.setChecked(false);

                return;
            }
            newAccelerator = Accelerator.GPU;
        } else  {
            newAccelerator = Accelerator.CPU;
            }
        if ( classifier.getAccelerator() != newAccelerator) {
            try {
                classifier.close();
                classifier = new Classifier(this, "mobilenetv2.tflite", "labels1.txt",IMAGE_SIZE, newAccelerator);
            }  catch (LiteRtException e) {
                Toast.makeText(this,
                        "Classifier konnte nicht erstellt werden",
                        Toast.LENGTH_SHORT
                ).show();
            }
        }
    }
        private void loadImage (Uri uri){
            try {
                selectedBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                selectedBitmap = rotateBitmapIfRequired(selectedBitmap, uri);

                imageView.setImageBitmap(selectedBitmap);


            } catch (IOException e) {
                e.printStackTrace();
                textViewResult.setText("Fehler beim Laden des Bildes!");
            }
        }

    private Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri imageUri) throws IOException {
        InputStream input = getContentResolver().openInputStream(imageUri);
        ExifInterface exif = new ExifInterface(input);
        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        input.close();

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
            default:
                return bitmap; // keine Rotation nötig
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    }
