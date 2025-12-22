package com.example.app2;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.AudioFormat;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.function.Consumer;
import org.tensorflow.lite.support.audio.TensorAudio;
public class MyAudioClassifier {
    private InterpreterApi interpreter;
    //Labels werden entweder per textdatei bereitgestellt, oder im Fall von bereitgestellten Modellen
    private List<String> labels;
    //vom Modell abhängig


    private MappedByteBuffer modelBuffer;

    public MyAudioClassifier(Context context, String modelFile, String labelsFile) {
        try {
            this.modelBuffer = FileUtil.loadMappedFile(context, modelFile);

            this.interpreter = InterpreterApi.create(
                    modelBuffer,
                    new InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            );
        } catch (IOException e) {
            throw new RuntimeException("TFLite Model Fehler!", e);
        }

        try {
            this.labels = FileUtil.loadLabels(context, labelsFile);
        } catch (IOException e) {
            throw new RuntimeException("Labels Datei nicht gefunden!", e);
        }
    }

    public void classify(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }
        int sampleRate = 16000; // 16 kHz
        float audioDuration = 1.0f; // 1



        int modelInputLength = (int)(sampleRate * audioDuration);
        TensorAudio t = TensorAudio.create()

        TensorAudio.TensorAudioFormat format = new TensorAudio.TensorAudioFormat() {
            @Override
            public int getChannels() {
                return 0;
            }

            @Override
            public int getSampleRate() {
                return 0;
            }
        };








        // --- Output ByteBuffer erstellen ---
        ByteBuffer outputByteBuffer = ByteBuffer.allocateDirect(labels.size()); // float pro Label
        outputByteBuffer.order(ByteOrder.nativeOrder());
        try {
            // Inferenz ausführen
            interpreter.run(mybytebuffer, outputByteBuffer);
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
}
