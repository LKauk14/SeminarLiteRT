package com.example.app2;

import android.graphics.Bitmap;
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

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.LiteRtException;

import java.io.IOException;

/**
 * @class MainActivity
 * @brief Hauptaktivität von App 2 (LiteRT mit Uint8-Modell).
 *
 * Diese Activity stellt die Benutzeroberfläche für die Bildklassifikation
 * mit einem quantisierten Uint8-Modell (MobileNetV1) bereit.
 *
 * Funktionen:
 * - Auswahl eines Bildes aus der Galerie
 * - Umschalten zwischen CPU- und GPU-Beschleunigung (falls unterstützt)
 * - Starten der Inferenz über die {@link Classifier}-Klasse
 *
 * Im Gegensatz zu App 3 wird hier:
 * - ein Uint8-Modell verwendet
 * - keine Bildrotation anhand von EXIF-Daten durchgeführt
 */
public class MainActivity extends AppCompatActivity {

    /** Eingabegröße des Modells (Breite = Höhe) */
    private static final int IMAGE_SIZE = 224;

    /** Button zum Auswählen eines Bildes */
    private Button buttonUpload;

    /** Button zum Starten der Klassifikation */
    private Button buttonClassify;

    /** ImageView zur Anzeige des geladenen Bildes */
    private ImageView imageView;

    /** TextView zur Anzeige der Klassifikationsergebnisse */
    private TextView textViewResult;

    /** Aktuell ausgewähltes Bild */
    private Bitmap selectedBitmap;

    /** Switch zur Auswahl des Accelerators (CPU/GPU) */
    private Switch switchAccelaratorMethod;

    /** Bildklassifikator (LiteRT) */
    private Classifier classifier;

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
     * Initialisiert UI, Classifier und Event-Listener.
     *
     * @param savedInstanceState gespeicherter Zustand der Activity
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Layout setzen
        setContentView(R.layout.activity_main);

        // UI Elemente initialisieren
        buttonUpload = findViewById(R.id.buttonUpload);
        buttonClassify = findViewById(R.id.buttonClassify);
        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        switchAccelaratorMethod = findViewById(R.id.switchAccelarator);

        // Listener für Accelerator-Umschaltung
        switchAccelaratorMethod.setOnCheckedChangeListener(this::onSwitchChanged);

        // Classifier initialisieren (Standard: CPU)
        try {
            classifier = new Classifier(
                    this,
                    "mobilenetv1.tflite",
                    "labels1.txt",
                    IMAGE_SIZE
            );
        } catch (LiteRtException e) {
            throw new RuntimeException(e);
        }

        // Button: Bild auswählen
        buttonUpload.setOnClickListener(v -> galleryLauncher.launch("image/*"));

        // Button: Klassifikation starten
        buttonClassify.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                textViewResult.setText("Bitte zuerst ein Bild auswählen!");
                return;
            }

            try {
                classifier.classify(selectedBitmap, result ->
                        runOnUiThread(() ->
                                textViewResult.setText("Ergebnis: " + result)
                        )
                );
            } catch (LiteRtException e) {
                textViewResult.setText("Fehler bei Inferenz: " + e.getMessage());
            }
        });
    }

    /**
     * Callback für den Accelerator-Switch.
     * Erstellt nur dann einen neuen {@link Classifier},
     * wenn sich der gewünschte Accelerator wirklich ändert.
     *
     * Falls GPU nicht verfügbar ist, bleibt der Switch im AUS-Zustand.
     *
     * @param buttonView Referenz auf den Switch
     * @param isChecked true = GPU, false = CPU
     */
    private void onSwitchChanged(CompoundButton buttonView, boolean isChecked) {

        Accelerator newAccelerator;

        if (isChecked) {
            // Prüfen, ob GPU unterstützt wird
            if (!classifier.isGpuSupported(this, "mobilenetv1.tflite")) {
                Toast.makeText(this, "GPU nicht verfügbar", Toast.LENGTH_SHORT).show();
                buttonView.setChecked(false);
                return;
            }
            newAccelerator = Accelerator.GPU;
        } else {
            newAccelerator = Accelerator.CPU;
        }

        // Classifier nur neu erstellen, wenn nötig
        if (classifier.getAccelerator() != newAccelerator) {
            try {
                classifier.close();
                classifier = new Classifier(
                        this,
                        "mobilenetv1.tflite",
                        "labels1.txt",
                        IMAGE_SIZE,
                        newAccelerator
                );
            } catch (LiteRtException e) {
                Toast.makeText(
                        this,
                        "Classifier konnte nicht erstellt werden",
                        Toast.LENGTH_SHORT
                ).show();
            }
        }
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
