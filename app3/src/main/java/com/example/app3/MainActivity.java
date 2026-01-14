package com.example.app3;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import androidx.exifinterface.media.ExifInterface;
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
import java.io.InputStream;

/**
 * @class MainActivity
 * @brief Hauptaktivität von App 3 (LiteRT mit CPU/GPU Umschaltung und Float modell).
 *
 * Diese Activity stellt die Benutzeroberfläche für die Bildklassifikation bereit.
 * Der Nutzer kann:
 * - ein Bild aus der Galerie auswählen
 * - zwischen CPU- und GPU-Beschleunigung wechseln
 * - eine Bildklassifikation starten
 *
 * Die Klassifikation erfolgt über die {@link Classifier}-Klasse,
 * welche ein LiteRT-Modell verwendet.
 *
 * Zusätzlich wird die Bildrotation anhand von EXIF-Daten korrigiert,
 * um falsche Klassifikationsergebnisse durch gedrehte Bilder zu vermeiden.
 */
public class MainActivity extends AppCompatActivity {

    /** Eingabegröße des Modells (Breite = Höhe) */
    private static final int IMAGE_SIZE = 224;

    /** Button zum Auswählen eines Bildes aus der Galerie */
    private Button buttonUpload;

    /** Button zum Starten der Klassifikation */
    private Button buttonClassify;

    /** ImageView zur Anzeige des ausgewählten Bildes */
    private ImageView imageView;

    /** TextView zur Anzeige der Klassifikationsergebnisse */
    private TextView textViewResult;

    /** Aktuell ausgewähltes Bild */
    private Bitmap selectedBitmap;

    /** Switch zur Auswahl des Accelerators (CPU/GPU) */
    private Switch switchAccelaratorMethod;

    /** Instanz des Bildklassifikators */
    private Classifier classifier;

    /**
     * ActivityResultLauncher zum Öffnen der Bildergalerie.
     * Nach Auswahl wird {@link #loadImage(Uri)} aufgerufen.
     */
    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    loadImage(uri);
                }
            });

    /**
     * Initialisiert die Benutzeroberfläche, den Classifier
     * und registriert alle Listener.
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

        // Listener für Accelerator-Switch
        switchAccelaratorMethod.setOnCheckedChangeListener(this::onSwitchChanged);

        // Classifier initialisieren (Standard: CPU)
        try {
            classifier = new Classifier(this, "mobilenetv2.tflite", "labels1.txt", IMAGE_SIZE);
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
                        runOnUiThread(() -> textViewResult.setText("Ergebnis: " + result))
                );
            } catch (LiteRtException e) {
                textViewResult.setText("Fehler bei Inferenz: " + e.getMessage());
            }
        });
    }

    /**
     * Callback für den Accelerator-Switch.
     * Wechselt zwischen CPU und GPU, falls GPU unterstützt wird.
     *
     * Falls GPU nicht verfügbar ist, bleibt der Switch im AUS-Zustand
     * und es wird kein neuer Classifier erstellt.
     *
     * @param buttonView Referenz auf den Switch
     * @param isChecked true = GPU, false = CPU
     */
    private void onSwitchChanged(CompoundButton buttonView, boolean isChecked) {

        Accelerator newAccelerator;

        if (isChecked) {
            // Prüfen, ob GPU unterstützt wird
            if (!classifier.isGpuSupported(this, "mobilenetv2.tflite")) {
                Toast.makeText(this, "GPU nicht verfügbar", Toast.LENGTH_SHORT).show();
                buttonView.setChecked(false);
                return;
            }
            newAccelerator = Accelerator.GPU;
        } else {
            newAccelerator = Accelerator.CPU;
        }

        // Nur neu erstellen, wenn sich der Accelerator wirklich ändert
        if (classifier.getAccelerator() != newAccelerator) {
            try {
                classifier.close();
                classifier = new Classifier(
                        this,
                        "mobilenetv2.tflite",
                        "labels1.txt",
                        IMAGE_SIZE,
                        newAccelerator
                );
            } catch (LiteRtException e) {
                Toast.makeText(this,
                        "Classifier konnte nicht erstellt werden",
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * Lädt ein Bild aus der Galerie, korrigiert die Ausrichtung
     * und zeigt es im ImageView an.
     *
     * @param uri URI des ausgewählten Bildes
     */
    private void loadImage(Uri uri) {
        try {
            selectedBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            selectedBitmap = rotateBitmapIfRequired(selectedBitmap, uri);
            imageView.setImageBitmap(selectedBitmap);
        } catch (IOException e) {
            e.printStackTrace();
            textViewResult.setText("Fehler beim Laden des Bildes!");
        }
    }

    /**
     * Korrigiert die Bildrotation anhand der EXIF-Orientierungsdaten.
     *
     * @param bitmap Ursprüngliches Bitmap
     * @param imageUri URI des Bildes
     * @return korrekt ausgerichtetes Bitmap
     * @throws IOException falls EXIF-Daten nicht gelesen werden können
     */
    private Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri imageUri) throws IOException {
        InputStream input = getContentResolver().openInputStream(imageUri);
        ExifInterface exif = new ExifInterface(input);
        int orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
        );
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

        return Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
    }
}
