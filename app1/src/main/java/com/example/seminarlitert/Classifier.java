package com.example.seminarlitert;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.function.Consumer;

/**
 * @class Classifier
 * @brief Bildklassifikator mit TensorFlow Lite Interpreter API (Google Play Services).
 *
 * <p>
 * Diese Klasse implementiert eine Bildklassifikation mithilfe der
 * {@link InterpreterApi} aus TensorFlow Lite.
 * </p>
 *
 * <p>
 * Besonderheiten dieser Implementierung:
 * <ul>
 *   <li>Verwendung der System-Runtime (Google Play Services)</li>
 *   <li>Keine GPU-/Delegate-Steuerung im Code möglich</li>
 *   <li>Manuelle ByteBuffer-Erstellung für Uint8-Modelle</li>
 * </ul>
 * </p>
 *
 * <p>
 * Unterstütztes Modellformat:
 * <ul>
 *   <li>Input: Uint8, Shape [1, H, W, 3]</li>
 *   <li>Output: Uint8, Shape [1, numLabels]</li>
 * </ul>
 * </p>
 */
public class Classifier {

    /** TensorFlow Lite Interpreter (System Runtime) */
    private InterpreterApi interpreter;

    /** Liste der Klassenlabels */
    private List<String> labels;

    /** Eingabegröße des Modells (z. B. 224) */
    private int imageSize;

    /** (Optional) aktuell ausgewähltes Bitmap */
    private Bitmap selectedBitmap;

    /** Memory-mapped TFLite-Modell */
    private MappedByteBuffer modelBuffer;

    /**
     * Konstruktor für Klassifikation mit externen Label-Dateien.
     *
     * @param context    Android Context
     * @param modelFile  TFLite-Modell im Assets-Ordner
     * @param labelsFile Label-Datei im Assets-Ordner
     * @param imageSize  Eingabegröße des Modells
     */
    public Classifier(Context context,
                      String modelFile,
                      String labelsFile,
                      int imageSize) {

        initInterpreter(context, modelFile);
        this.imageSize = imageSize;

        try {
            this.labels = FileUtil.loadLabels(context, labelsFile);
        } catch (IOException e) {
            throw new RuntimeException("Labels Datei nicht gefunden!", e);
        }
    }

    /**
     * Initialisiert den TensorFlow Lite Interpreter mit der System-Runtime.
     *
     * <p>
     * Es wird explizit {@link TfLiteRuntime#FROM_SYSTEM_ONLY} verwendet,
     * wodurch das Modell über Google Play Services ausgeführt wird.
     * </p>
     *
     * @param context   Android Context
     * @param modelFile Modell-Dateiname im Assets-Ordner
     */
    private void initInterpreter(Context context, String modelFile) {
        try {
            this.modelBuffer = FileUtil.loadMappedFile(context, modelFile);

            this.interpreter = InterpreterApi.create(
                    modelBuffer,
                    new InterpreterApi.Options()
                            .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
            );
        } catch (IOException e) {
            throw new RuntimeException("TFLite Model Fehler!", e);
        }
    }

    /**
     * Führt eine Bildklassifikation mit manuell erzeugtem ByteBuffer aus.
     *
     * <p>
     * Die Pixelwerte werden als RGB-Bytes (Uint8) in der Reihenfolge
     * [R, G, B] gespeichert und an den Interpreter übergeben.
     * </p>
     *
     * @param bitmap   Eingabebild
     * @param callback Callback zur Ausgabe des Klassifikationsergebnisses
     */
    public void classify(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }

        // Eingabe-Tensor-Shape ermitteln
        int[] shape = interpreter.getInputTensor(0).shape();
        int INPUT_HEIGHT = shape[1];
        int INPUT_WIDTH = shape[2];

        // Bitmap auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(
                bitmap, INPUT_WIDTH, INPUT_HEIGHT, true
        );

        // ByteBuffer für Uint8-RGB-Daten
        ByteBuffer inputBuffer =
                ByteBuffer.allocateDirect(INPUT_HEIGHT * INPUT_WIDTH * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_WIDTH * INPUT_HEIGHT];
        scaled.getPixels(
                pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT
        );

        // Pixel von ARGB → RGB (Uint8)
        int pixelIndex = 0;
        for (int y = 0; y < INPUT_HEIGHT; y++) {
            for (int x = 0; x < INPUT_WIDTH; x++) {
                int pixel = pixels[pixelIndex++];
                inputBuffer.put((byte) ((pixel >> 16) & 0xFF)); // R
                inputBuffer.put((byte) ((pixel >> 8) & 0xFF));  // G
                inputBuffer.put((byte) (pixel & 0xFF));         // B
            }
        }

        inputBuffer.rewind();

        // Output-Buffer (Uint8)
        ByteBuffer outputBuffer =
                ByteBuffer.allocateDirect(labels.size());
        outputBuffer.order(ByteOrder.nativeOrder());

        try {
            interpreter.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();

            byte[] rawOutput = new byte[labels.size()];
            outputBuffer.get(rawOutput);

            // Output in Wahrscheinlichkeiten umwandeln
            float[] probs = new float[labels.size()];
            for (int i = 0; i < rawOutput.length; i++) {
                probs[i] = (rawOutput[i] & 0xFF) / 255.0f;
            }

            // Top-1 Ergebnis bestimmen
            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            callback.accept(
                    labels.get(maxIndex) +
                            String.format(" (%.2f%%)", maxProb * 100)
            );

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }

    /**
     * Alternative Klassifikation unter Nutzung von {@link TensorImage}.
     *
     * <p>
     * Diese Methode verwendet die TensorFlow Lite Support Library
     * zur Bildvorverarbeitung (Resize, Formatierung).
     * </p>
     *
     * @param bitmap   Eingabebild
     * @param callback Callback für das Klassifikationsergebnis
     */
    public void classifyWithTensorImage(Bitmap bitmap,
                                        Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(
                                imageSize,
                                imageSize,
                                ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        TensorImage tensorImage =
                imageProcessor.process(TensorImage.fromBitmap(bitmap));

        ByteBuffer inputBuffer = tensorImage.getBuffer();

        ByteBuffer outputBuffer =
                ByteBuffer.allocateDirect(labels.size());
        outputBuffer.order(ByteOrder.nativeOrder());

        try {
            interpreter.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();

            float[] probs = new float[labels.size()];
            for (int i = 0; i < probs.length; i++) {
                probs[i] = (outputBuffer.get() & 0xFF) / 255f;
            }

            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            callback.accept(
                    labels.get(maxIndex) +
                            String.format(" (%.2f%%)", maxProb * 100)
            );

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }
}
