package com.example.seminarlitert;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import android.content.Context;
import android.graphics.Bitmap;

import com.google.android.gms.tasks.Task;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class Classifier {
    private InterpreterApi interpreter;
    //Labels werden entweder per textdatei bereitgestellt, oder im Fall von bereitgestellten Modellen
    private List<String> labels;
    //vom Modell abhängig
    private int imageSize;

    private MappedByteBuffer modelBuffer;

    //wenn kein Label enthalten -> ziehe aus Modell Metadaten
    // auch möglich: prüfe, ob metadaten labels enthalten. Wenn ja, nutze es wenn nein frage ab und lies File von User ein
    public Classifier(Context context, String modelFile, int imageSize) {
        initInterpreter(context, modelFile);
        this.imageSize = imageSize;

        try {
            this.labels = loadEmbeddedLabels(modelBuffer);
        } catch (IOException e) {
            throw new RuntimeException("Keine embedded labels gefunden!", e);
        }
    }


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
            modelBuffer = FileUtil.loadMappedFile(context, modelFile);

            interpreter = InterpreterApi.create(
                    modelBuffer,
                    new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
            );
        } catch (IOException e) {
            throw new RuntimeException("TFLite Model Fehler!", e);
        }
    }

    //labels im Modell
    private List<String> loadEmbeddedLabels(MappedByteBuffer modelBuffer) throws IOException {

        MetadataExtractor extractor = new MetadataExtractor(modelBuffer);

        InputStream labelsInput = extractor.getAssociatedFile("labels.txt");

        List<String> result = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(labelsInput));

        String line;
        while ((line = reader.readLine()) != null) {
            result.add(line);
        }

        return result;
    }

    public void classify(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }

        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);

        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(
                new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32
        );

        float[] inputData = new float[imageSize * imageSize * 3];
        int idx = 0;

        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {

                int pixel = scaled.getPixel(x, y);

                inputData[idx++] = ((pixel >> 16) & 0xFF) / 255f;
                inputData[idx++] = ((pixel >> 8) & 0xFF) / 255f;
                inputData[idx++] = (pixel & 0xFF) / 255f;
            }
        }

        inputBuffer.loadArray(inputData);

        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
                new int[]{1, labels.size()}, DataType.FLOAT32
        );


        //Task<Void> task;
        //task = interpreter.run(inputBuffer, outputBuffer);
        //synchrone lösung mit interpreter api -> bei laufender UI sollte je nachdem async sein
        try {
            //Inferenz
            interpreter.run(inputBuffer, outputBuffer);

            float[] probs = outputBuffer.getFloatArray();

            // Top-1 Label berechnen
            int maxIndex = 0;
            float maxProb = 0;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            callback.accept(labels.get(maxIndex) + String.format(" (%.2f%%)", maxProb * 100));

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }

}