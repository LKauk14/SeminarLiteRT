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

/**
 * @class MainActivity
 * @brief Hauptaktivität von App 1 (Interpreter API).
 *
 * Diese Activity ermöglicht:
 * - Auswahl eines Bildes aus der Galerie
 * - Auswahl der Klassifikationsmethode (NORMAL / TENSOR_IMAGE)
 * - Starten der Inferenz über die {@link Classifier}-Klasse
 *
 * Besonderheiten:
 * - Unterstützt zwei Klassifikationsmodi:
 *   - NORMAL: klassische ByteBuffer-Input-Konvertierung
 *   - TENSOR_IMAGE: TensorImage + ImageProcessor Pipeline
 * - Nutzt die TensorFlow Lite Interpreter API (InterpreterApi)
 * - Labels werden aus "labels.txt" geladen
 */
public class MainActivity extends AppCompatActivity {

    /** Mögliche Klassifikationsmodi */
    enum ClassifyMode {
        NORMAL,
        TENSOR_IMAGE
    }

    /** Request-Code für die Galerie (nicht zwingend genutzt, aber konventionell) */
    private static final int IMAGE_PICK_CODE = 1001;

    /** Eingabegröße des Modells (Breite = Höhe) */
    private static final int IMAGE_SIZE = 224;

    /** TensorFlow Lite Interpreter (Interpreter API) */
    private InterpreterApi interpreter;

    /** Bildklassifikator */
    private Classifier classifier;

    /** Aktuell ausgewählter Klassifikationsmodus */
    ClassifyMode mode = ClassifyMode.NORMAL;

    /** UI Elemente */
    private Button buttonClassify;
    private Button buttonUpload;
    private ImageView imageView;
    private TextView textViewResult;
    private Bitmap selectedBitmap;
    private Switch switchMethod;

    /**
     * Launcher zum Öffnen der System-Galerie.
     * Nach Auswahl eines Bildes wird {@link #loadImage(Uri)} aufgerufen.
     */
    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    loadImage(uri);
                }
            });

    /**
     * Initialisiert UI, Klassifikator und Event-Listener.
     *
     * @param savedInstanceState gespeicherter Zustand der Activity
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Layout setzen
        setContentView(R.layout.activity_main);

        // UI Elemente initialisieren
        switchMethod = findViewById(R.id.switchMethod);
        imageView = findViewById(R.id.imageView);
        buttonUpload = findViewById(R.id.buttonUpload);
        buttonClassify = findViewById(R.id.buttonClassify);
        textViewResult = findViewById(R.id.textViewResult);

        // TensorFlow Lite Interpreter initialisieren
        Task<Void> initializeTask = TfLite.initialize(this);
        initializeTask.addOnSuccessListener(a -> {
                    classifier = new Classifier(this, "mobilenetv1.tflite", "labels.txt", IMAGE_SIZE);
                })
                .addOnFailureListener(e -> {
                    Log.e("Interpreter", String.format(
                            "Cannot initialize interpreter: %s", e.getMessage()));
                });

        // Button: Bild auswählen
        buttonUpload.setOnClickListener(v -> pickImageFromGallery());

        // Button: Klassifikation starten
        buttonClassify.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                textViewResult.setText("Bitte zuerst ein Bild auswählen!");
                return;
            }

            // Klassifikationsmodus auswählen
            mode = switchMethod.isChecked() ? ClassifyMode.TENSOR_IMAGE : ClassifyMode.NORMAL;

            switch (mode) {
                case NORMAL:
                    classifier.classify(selectedBitmap, result -> runOnUiThread(() ->
                            textViewResult.setText("Ergebnis: " + result)
                    ));
                    break;
                case TENSOR_IMAGE:
                    classifier.classifyWithTensorImage(selectedBitmap, result -> runOnUiThread(() ->
                            textViewResult.setText("Ergebnis: " + result)
                    ));
                    break;
            }
        });
    }

    /**
     * Startet den Galerie-Picker.
     */
    private void pickImageFromGallery() {
        galleryLauncher.launch("image/*");
    }

    /**
     * Lädt ein Bild aus der Galerie und zeigt es im ImageView an.
     *
     * @param uri URI des ausgewählten Bildes
     */
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
