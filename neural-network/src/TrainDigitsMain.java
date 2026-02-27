import java.util.Random;

public class TrainDigitsMain {
    public static void main(String[] args) throws Exception {

        // Mets ici TES chemins (gz ou non, les deux marchent)
        String trainImages = "/home/kaine/Downloads/mnist/train-images-idx3-ubyte.gz";
        String trainLabels = "/home/kaine/Downloads/mnist/train-labels-idx1-ubyte.gz";

        // Optionnel (pour tester vite l'accuracy)
        String testImages  = "/home/kaine/Downloads/mnist/t10k-images-idx3-ubyte.gz";
        String testLabels  = "/home/kaine/Downloads/mnist/t10k-labels-idx1-ubyte.gz";

        int trainLimit = 60000; // mets 10000 pour tester plus vite
        int testLimit  = 10000;

        System.out.println("Loading MNIST...");
        DataPoint[] train = MnistReader.load(trainImages, trainLabels, trainLimit);
        DataPoint[] test  = MnistReader.load(testImages, testLabels, testLimit);

        NeuralNetwork nn = new NeuralNetwork(784, 128, 64, 10);

        double learnRate = 1.0;
        int epochs = 15;
        int batchSize = 64;  // 32/64/128

        Random rng = new Random(123);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffle(train, rng);

            for (int i = 0; i < train.length; i += batchSize) {
                int end = Math.min(train.length, i + batchSize);
                DataPoint[] batch = new DataPoint[end - i];
                System.arraycopy(train, i, batch, 0, batch.length);

                nn.Learn(batch, learnRate);
            }

            double acc = accuracy(nn, test);
            System.out.println("Epoch " + epoch + " done | test accuracy: " + String.format("%.2f", acc * 100) + "%");
        }

        String outPath = "/home/kaine/Documents/projets-perso/neural-network/digits_model.txt";
        nn.save(outPath);
        System.out.println("SAVED: " + outPath);
    }

    static void shuffle(DataPoint[] arr, Random rng) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            DataPoint tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    static double accuracy(NeuralNetwork nn, DataPoint[] data) {
        int correct = 0;
        for (DataPoint dp : data) {
            int pred = nn.Classify(dp.inputs);
            int expected = argmax(dp.expectedOutputs);
            if (pred == expected) correct++;
        }
        return correct / (double) data.length;
    }

    static int argmax(double[] a) {
        int mi = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[mi]) mi = i;
        }
        return mi;
    }
}
