package com.example.seminarlitert;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;


import com.google.android.gms.tasks.Task;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.function.Consumer;

public class Classifier {
    private InterpreterApi interpreter;
    //Labels werden entweder per textdatei bereitgestellt, oder im Fall von bereitgestellten Modellen
    private List<String> labels;
    //vom Modell abhängig
    private int imageSize;
    private Bitmap selectedBitmap;
    private MappedByteBuffer modelBuffer;

    //wenn kein Label enthalten -> ziehe aus Modell Metadaten
    // auch möglich: prüfe, ob metadaten labels enthalten. Wenn ja, nutze es wenn nein frage ab und lies File von User ein

   /* public Classifier(Context context, String modelFile, int imageSize) {
        initInterpreter(context, modelFile);
        this.imageSize = imageSize;

        try {
            this.labels = loadEmbeddedLabels(modelBuffer);
        } catch (IOException e) {
            throw new RuntimeException("Keine embedded labels gefunden!", e);
        }
    }
*/

    // wenn Labels übergeben, nutze File (bei eigenen Modellen);
    public Classifier(Context context, String modelFile, String labelsFile, int imageSize) {
        initInterpreter(context, modelFile);
        this.imageSize = imageSize;

        try {
            this.labels = FileUtil.loadLabels(context, labelsFile);
        } catch (IOException e) {
            throw new RuntimeException("Labels Datei nicht gefunden!", e);
        }
    }


    private void initInterpreter(Context context, String modelFile) {
        try {
            this.modelBuffer = FileUtil.loadMappedFile(context, modelFile);

            this.interpreter = InterpreterApi.create(
                    modelBuffer,
                    new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
            );
        } catch (IOException e) {
            throw new RuntimeException("TFLite Model Fehler!", e);
        }
    }

    public void classify(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }

        int[] shape = interpreter.getInputTensor(0).shape();
        int batch = shape[0];
        int INPUT_HEIGHT = shape[1];
        int INPUT_WIDTH = shape[2];
        int channels = shape[3];

        // Bild auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, INPUT_HEIGHT, INPUT_WIDTH, true);
        ByteBuffer byteBufferConvertTest = ByteBuffer.allocateDirect(1*INPUT_HEIGHT*INPUT_WIDTH*3);
        byteBufferConvertTest.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_WIDTH * INPUT_HEIGHT];
        scaled.getPixels(
                pixels,
                0,
                INPUT_WIDTH,
                0,
                0,
                INPUT_WIDTH,
                INPUT_HEIGHT
        );

        int pixelIndex = 0;
        for (int y = 0; y < INPUT_HEIGHT; y++) {
            for (int x = 0; x < INPUT_WIDTH; x++) {
                int pixel = pixels[pixelIndex++];

                // RGB extrahieren (0–255)
                byte r = (byte) ((pixel >> 16) & 0xFF);
                byte g = (byte) ((pixel >> 8) & 0xFF);
                byte b = (byte) (pixel & 0xFF);
                byteBufferConvertTest.put(r);
                byteBufferConvertTest.put(g);
                byteBufferConvertTest.put(b);
            }
        }

        byteBufferConvertTest.rewind();

        // --- Output ByteBuffer erstellen ---
        ByteBuffer outputByteBuffer = ByteBuffer.allocateDirect(labels.size()); // float pro Label
        outputByteBuffer.order(ByteOrder.nativeOrder());

        try {
            // Inferenz ausführen
            interpreter.run(byteBufferConvertTest, outputByteBuffer);
            outputByteBuffer.rewind();

            byte[] rawOutput = new byte[labels.size()];
            outputByteBuffer.get(rawOutput);

            float[] probs = new float[labels.size()];
            for (int i = 0; i < rawOutput.length; i++) {
                probs[i] = (rawOutput[i] & 0xFF) / 255.0f;
            }

            // Top-1 Label bestimmen
            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }


            // Ergebnis zurückgeben
            callback.accept(labels.get(maxIndex) + String.format(" (%.2f%%)", maxProb * 100));

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }

    public void classifyWithTensorImage(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }

        // Bild auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder().add(new ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR)).build();

        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap));
        ByteBuffer inputBuffer = tensorImage.getBuffer();

        // --- Output ByteBuffer erstellen ---
        ByteBuffer outputByteBuffer = ByteBuffer.allocateDirect(labels.size()); // float pro Label
        outputByteBuffer.order(ByteOrder.nativeOrder());

        try {
            // Inferenz ausführen
            interpreter.run(inputBuffer, outputByteBuffer);
            outputByteBuffer.rewind();

            float[] probs = new float[labels.size()];

            for (int i = 0; i < probs.length; i++) {
                probs[i] = (outputByteBuffer.get() & 0xFF) / 255f;
            }

            // Top-1 Label bestimmen
            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            // Ergebnis zurückgeben
            callback.accept(labels.get(maxIndex) + String.format(" (%.2f%%)", maxProb * 100));

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }
}