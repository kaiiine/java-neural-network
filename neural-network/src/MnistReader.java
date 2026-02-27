import java.io.*;
import java.util.zip.GZIPInputStream;

public class MnistReader {

    public static DataPoint[] load(String imagesPath, String labelsPath, int limit) throws IOException {
        try (DataInputStream img = new DataInputStream(new BufferedInputStream(open(imagesPath)));
             DataInputStream lbl = new DataInputStream(new BufferedInputStream(open(labelsPath)))) {

            int magicImages = img.readInt();
            int numImages = img.readInt();
            int rows = img.readInt();
            int cols = img.readInt();

            int magicLabels = lbl.readInt();
            int numLabels = lbl.readInt();

            if (magicImages != 2051) throw new IOException("Bad MNIST images magic: " + magicImages);
            if (magicLabels != 2049) throw new IOException("Bad MNIST labels magic: " + magicLabels);
            if (rows != 28 || cols != 28) throw new IOException("Expected 28x28, got " + rows + "x" + cols);

            int n = Math.min(Math.min(numImages, numLabels), limit);
            DataPoint[] data = new DataPoint[n];

            for (int i = 0; i < n; i++) {
                double[] inputs = new double[28 * 28];
                for (int p = 0; p < inputs.length; p++) {
                    int b = img.readUnsignedByte();
                    inputs[p] = b / 255.0;
                }

                int label = lbl.readUnsignedByte();
                double[] expected = new double[10];
                expected[label] = 1.0;

                data[i] = new DataPoint(inputs, expected);
            }

            return data;
        }
    }

    private static InputStream open(String path) throws IOException {
        InputStream in = new FileInputStream(path);
        if (path.endsWith(".gz")) {
            return new GZIPInputStream(in);
        }
        return in;
    }
}
