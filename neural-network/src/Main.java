import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) {

        DataPoint[] xor = new DataPoint[]{
                new DataPoint(new double[]{0, 0}, new double[]{0}),
                new DataPoint(new double[]{0, 1}, new double[]{1}),
                new DataPoint(new double[]{1, 0}, new double[]{1}),
                new DataPoint(new double[]{1, 1}, new double[]{0})
        };

        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

        System.out.println("Avant apprentissage:");
        printPredictions(nn, xor);
        System.out.println("Cost: " + nn.Cost(xor));

        double learnRate = 1.0;
        int epochs = 20000;
        Random rng = new Random();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffleInPlace(xor, rng);
            nn.Learn(xor, learnRate);

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + " | Cost: " + nn.Cost(xor));
            }
        }

        System.out.println("\nAprès apprentissage:");
        printPredictions(nn, xor);
        System.out.println("Cost: " + nn.Cost(xor));

        System.out.println("\nTest binaire (seuil 0.5):");
        testBinary(nn);

        String path = "/home/kaine/Documents/projets-perso/neural-network/xor_model.txt";

        // 1) Sauvegarde
        try {
            nn.save(path);
            System.out.println("\nSAVED: " + path);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // 2) Recharge et test
        try {
            NeuralNetwork nnReloaded = NeuralNetwork.load(path);
            System.out.println("\n=== TEST APRÈS RELOAD ===");
            printPredictions(nnReloaded, xor);
            System.out.println("Cost reload: " + nnReloaded.Cost(xor));
            System.out.println("\nTest binaire après reload:");
            testBinary(nnReloaded);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static void printPredictions(NeuralNetwork nn, DataPoint[] data) {
        for (DataPoint dp : data) {
            double[] out = nn.CalculateOutputs(dp.inputs);
            System.out.println(Arrays.toString(dp.inputs)
                    + " -> predicted=" + Arrays.toString(out)
                    + " expected=" + Arrays.toString(dp.expectedOutputs));
        }
    }

    static void testBinary(NeuralNetwork nn) {
        testOne(nn, new double[]{0, 0});
        testOne(nn, new double[]{0, 1});
        testOne(nn, new double[]{1, 0});
        testOne(nn, new double[]{1, 1});
    }

    static void testOne(NeuralNetwork nn, double[] in) {
        double out = nn.CalculateOutputs(in)[0];
        int cls = out >= 0.5 ? 1 : 0;
        System.out.println(Arrays.toString(in) + " -> " + out + " => " + cls);
    }

    static void shuffleInPlace(DataPoint[] arr, Random rng) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            DataPoint tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
}
