package com.example.seminarlitert;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.Rot90Op;

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

    //labels im Modell-> über google play services nicht möglich
    /*
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
    }*/

    public void classify(Bitmap bitmap, Consumer<String> callback) {

        if (interpreter == null) {
            callback.accept("Interpreter nicht initialisiert");
            return;
        }
        // Bild auf Modellgröße skalieren
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        ByteBuffer byteBufferConvertTest = convertBitmapToByteArray(scaled);




        TensorImage MyTensorImage = new TensorImage((DataType.UINT8));
        MyTensorImage.load(scaled);

        ByteBuffer mybytebuffer = MyTensorImage.getBuffer();







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

    public static ByteBuffer convertBitmapToByteArray(Bitmap bitmap){
        ByteBuffer byteBuffer = ByteBuffer.allocate(bitmap.getByteCount());
        bitmap.copyPixelsToBuffer(byteBuffer);
        byteBuffer.rewind();
        return byteBuffer;
    }
}