package com.example.app2;

import android.content.Context;
import android.graphics.Bitmap;

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    // wenn Labels übergeben, nutze File (bei eigenen Modellen);
    public Classifier(Context context, String modelFile, String labelsFile,int imageSize) throws LiteRtException {
        compiledModel =
                CompiledModel.create(
                        context.getAssets(),
                        modelFile,
                        new CompiledModel.Options(Accelerator.CPU),
                        null
                );

        this.imageSize = imageSize;
        this.labels = loadLabels(context, "labels1.txt");
    }


    public void classify(Bitmap bitmap, Consumer<String> callback) throws LiteRtException {

        List<TensorBuffer> inputBuffers = compiledModel.createInputBuffers();
        List<TensorBuffer> outputBuffers = compiledModel.createOutputBuffers();

        // Bild auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        byte[] input = new byte[imageSize * imageSize * 3]; // 3 Kanäle: R,G,B
        int[] pixels = new int[imageSize * imageSize];
        scaled.getPixels(
                pixels,0,imageSize,0,0,imageSize,imageSize);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            input[i * 3]     = (byte)((pixel >> 16) & 0xFF); // R
            input[i * 3 + 1] = (byte)((pixel >> 8) & 0xFF);  // G
            input[i * 3 + 2] = (byte)(pixel & 0xFF);         // B
        }
        inputBuffers.get(0).writeInt8(input);


        try {
           compiledModel.run(inputBuffers, outputBuffers); // Inferenz ausführen
            TensorBuffer outputTensor = outputBuffers.get(0);
            byte[] outputBuffer = outputTensor.readInt8();

            float[] probabilities = new float[outputBuffer.length];
            for (int i = 0; i < outputBuffer.length; i++) {
                probabilities[i] = (outputBuffer[i] & 0xFF) / 255.0f; // Byte → unsigned → float
            }

            int topClass = 0;
            float topScore = probabilities[0];

            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > topScore) {
                    topScore = probabilities[i];
                    topClass = i;
                }
            }



            // Ergebnis zurückgeben
            callback.accept(labels.get(topClass) + String.format(" (%.2f%%)", topScore * 100));

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

  /*  public void classifyWithTensorImage(Bitmap bitmap, Consumer<String> callback) {

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
    }*/
}