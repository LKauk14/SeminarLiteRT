package com.example.app3;

import android.content.Context;
import android.graphics.Bitmap;

import android.widget.Toast;

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class Classifier {

    //Labels werden entweder per textdatei bereitgestellt, oder im Fall von bereitgestellten Modellen
    private List<String> labels;
    //vom Modell abhängig
    private int imageSize;
    private Bitmap selectedBitmap;
    private MappedByteBuffer modelBuffer;
    private CompiledModel compiledModel;
    private Environment env;
    private Accelerator accelerator;


    // wenn Labels übergeben, nutze File (bei eigenen Modellen);
    public Classifier(Context context, String modelFile, String labelsFile,int imageSize) throws LiteRtException {
       try{
           this.accelerator = Accelerator.CPU;

        compiledModel =
                CompiledModel.create(
                        context.getAssets(),
                        modelFile,
                        new CompiledModel.Options(accelerator),
                        null
                );

        this.imageSize = imageSize;
        this.labels = loadLabels(context, labelsFile);
    } catch (LiteRtException e) {
           Toast.makeText(context, "Fehler beim Laden des Modells: " + e.getMessage(), Toast.LENGTH_LONG).show();
           e.printStackTrace();
       }
       catch (Exception e) {
           throw new RuntimeException(e);
       }
    }

    public Classifier(Context context, String modelFile,String labelsFile , int imageSize, Accelerator newAccelerator) throws LiteRtException {

        try {
            this.accelerator = newAccelerator;
            compiledModel =
                    CompiledModel.create(
                            context.getAssets(),
                            modelFile,
                            new CompiledModel.Options(accelerator),
                            null
                    );

            this.imageSize = imageSize;
            this.labels = loadLabels(context, labelsFile);
        } catch (LiteRtException e){
            Toast.makeText(context, "Fehler beim Laden des Modells: " + e.getMessage(), Toast.LENGTH_LONG).show();
            e.printStackTrace();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void close() throws LiteRtException {
        compiledModel.close();
    }

    public void classify(Bitmap bitmap, Consumer<String> callback) throws LiteRtException {

        List<TensorBuffer> inputBuffers = compiledModel.createInputBuffers();
        List<TensorBuffer> outputBuffers = compiledModel.createOutputBuffers();

        // Bild auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        float[] input = new float[imageSize * imageSize * 3];
        int[] pixels = new int[imageSize * imageSize];
        scaled.getPixels(
                pixels,0,imageSize,0,0,imageSize,imageSize);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // Pixel normalisieren: -1 bis 1
            float r = ((pixel >> 16) & 0xFF) / 127.5f - 1f;
            float g = ((pixel >> 8) & 0xFF) / 127.5f - 1f;
            float b = (pixel & 0xFF) / 127.5f - 1f;

            input[i * 3]     = r;
            input[i * 3 + 1] = g;
            input[i * 3 + 2] = b;
        }
        inputBuffers.get(0).writeFloat(input);

        try {
            long startTime = System.nanoTime();
            compiledModel.run(inputBuffers, outputBuffers); // Inferenz ausführen
            long endTime = System.nanoTime();
            long durationMs = (endTime - startTime) / 1_000_000;
            TensorBuffer outputTensor = outputBuffers.get(0);
            float[] outputArray = outputTensor.readFloat();


            int[] topIndices = new int[3];
            float[] topProbs = new float[3];

            for (int i = 0; i < outputArray.length; i++) {
                float p = outputArray[i];
                for (int j = 0; j < 3; j++) {
                    if (p > topProbs[j]) {
                        for (int k = 2; k > j; k--) {
                            topProbs[k] = topProbs[k-1];
                            topIndices[k] = topIndices[k-1];
                        }
                        topProbs[j] = p;
                        topIndices[j] = i;
                        break;
                    }
                }
            }


            StringBuilder result = new StringBuilder("Top 3:\n");
            for (int i = 0; i < 3; i++) {
                result.append(labels.get(topIndices[i]))
                        .append(String.format(" (%.2f%%)", topProbs[i] * 100))
                        .append("\n");
            }
            result.append("Inferenzzeit: ").append(durationMs).append("ms");

            callback.accept(result.toString());

        } catch (Exception e) {
            callback.accept("Fehler bei Inference: " + e.getMessage());
        }
    }

    public List<String> loadLabels(Context context, String fileName) {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(fileName)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;
    }



    public boolean isGpuSupported(Context context, String modelFile) {
        try {
            CompiledModel gpuTest = CompiledModel.create(context.getAssets(), modelFile, new CompiledModel.Options(Accelerator.GPU));
            gpuTest.close(); // danach wieder schließen
            return true; // GPU möglich
        } catch (Exception e) {
            return false; // GPU nicht möglich
        }
    }

    public Accelerator getAccelerator() {
        return accelerator;
    }



}