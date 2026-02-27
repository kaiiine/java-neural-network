import java.util.List;
import java.util.ArrayList;
import java.io.*;
import java.util.Locale;

public class NeuralNetwork {

    private Layer[] layers;

    public NeuralNetwork(int... layerSizes){
        this.layers = new Layer[layerSizes.length-1];
        for (int i = 0; i < (layerSizes.length-1); i++) {
            this.layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }

        for (Layer layer : this.layers) {
            layer.InitializeRandomWeights();
        }
    }

    // Run the input values through the neural network to calculate the output values
    public double[] CalculateOutputs(double[] inputs){
        for(Layer layer : this.layers){
            inputs = layer.caclulOutputs(inputs);
        }
        return inputs;
    }


    // Run the inputs through the neural network and calculate which output node has the highest value
    public int Classify(double[] inputs){
        double[] outputs = CalculateOutputs(inputs);
        return indexOfMax(outputs);
    }

    public static int indexOfMax(double[] arr){
        int maxIndex = 0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] > arr[maxIndex]){
                maxIndex=i;
            }
        }
        return  maxIndex;
    }

    public static int[] indexOfMaxValues(double[] arr) {
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == max) {
                indices.add(i);
            }
        }

        // New integer array to transform indices array from a List array to an int[] array
        int[] final_array = new int[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            final_array[i]=indices.get(i);
        }
        return final_array;
    }

    // Loss function
    public double Cost(DataPoint datapoint){
        double[] outputs = CalculateOutputs(datapoint.inputs);
        Layer outputLayer = this.layers[this.layers.length - 1];
        double cost = 0;

        for(int nodeOut=0; nodeOut<outputs.length; nodeOut++){
            cost += outputLayer.nodeCost(outputs[nodeOut], datapoint.expectedOutputs[nodeOut]);
        }
        return cost;
    }
    public double Cost(DataPoint[] datas){
        double total_cost = 0;
        for(DataPoint datapoint : datas){
            total_cost+=Cost(datapoint);
        }
        return total_cost / datas.length;
    }


    public void OldLearn(DataPoint[] trainingData, double learnRate){
        final double h = 0.0001;
        double originalCost = Cost(trainingData);

        for (Layer layer : this.layers){
            // Calculate the cost gradient for the current weights
            for (int nodeIn=0; nodeIn<layer.getNumNodesIn(); nodeIn++){
                for (int nodeOut=0; nodeOut<layer.getNumNodesOut(); nodeOut++){
                    layer.getWeights()[nodeIn][nodeOut] += h;
                    double deltaCost = Cost(trainingData) - originalCost;
                    layer.getWeights()[nodeIn][nodeOut] -= h;
                    layer.getCostGradientW()[nodeIn][nodeOut] = deltaCost / (h);
                }
            }

            // Calculate the cost gradient for the current biases
            for (int biasIndex=0; biasIndex<layer.getBiases().length; biasIndex++){
                layer.getBiases()[biasIndex] += h;
                double deltaCost = Cost(trainingData) - originalCost;
                layer.getBiases()[biasIndex] -= h;
                layer.getCostGradientB()[biasIndex] = deltaCost / (h);
            }
        }
        // Call AppliGradient() on all layers
        for (int i = 0; i < this.layers.length; i++) {
            this.layers[i].ApplyGradients(learnRate/trainingData.length);
        }
    }

    public void Learn(DataPoint[] trainingBatch, double learnRate){
        for (DataPoint datapoint : trainingBatch){
            UpdateAllGradients(datapoint);
        }

        ApplyAllGradient(learnRate/trainingBatch.length);

        ClearAllGradients();

    }

    public void UpdateAllGradients(DataPoint datapoint){
        CalculateOutputs(datapoint.inputs);
        Layer outputlayer = this.layers[this.layers.length-1];
        double[] nodeValues = outputlayer.CalculateOutputLayerNodeValue(datapoint.expectedOutputs);
        outputlayer.UpdateGradient(nodeValues);

        // Loop backwards through all the hidden layers and update their gradients
        for (int hiddenLayerIndex=(this.layers.length-2); hiddenLayerIndex>=0; hiddenLayerIndex--){
            Layer hiddenLayer = this.layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(this.layers[hiddenLayerIndex+1], nodeValues);
            hiddenLayer.UpdateGradient(nodeValues);
        }

    }

    public void ApplyAllGradient(double learnRate){
        for (Layer layer : layers){
            layer.ApplyGradients(learnRate);
        }
    }

    public void ClearAllGradients(){
        for (Layer layer : layers){
            layer.ClearGradients();
        }
    }


    public void save(String path) throws IOException {
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {

            // write architecture
            out.print(layers[0].getNumNodesIn());
            for (Layer layer : layers) {
                out.print(" " + layer.getNumNodesOut());
            }
            out.println();

            // write params
            for (Layer layer : layers) {
                double[][] w = layer.getWeights();
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        out.print(String.format(Locale.US, "%.17g", w[i][j]));
                        out.print(" ");
                    }
                }
                out.println();

                double[] b = layer.getBiases();
                for (int j = 0; j < b.length; j++) {
                    out.print(String.format(Locale.US, "%.17g", b[j]));
                    out.print(" ");
                }
                out.println();
            }
        }
    }

    public static NeuralNetwork load(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String[] sizeParts = br.readLine().trim().split("\\s+");
            int[] sizes = new int[sizeParts.length];
            for (int i = 0; i < sizeParts.length; i++) {
                sizes[i] = Integer.parseInt(sizeParts[i]);
            }

            NeuralNetwork nn = new NeuralNetwork(sizes);

            for (Layer layer : nn.layers) {
                String[] wParts = br.readLine().trim().split("\\s+");
                int idx = 0;
                double[][] w = layer.getWeights();
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        w[i][j] = Double.parseDouble(wParts[idx++]);
                    }
                }

                String[] bParts = br.readLine().trim().split("\\s+");
                double[] b = layer.getBiases();
                for (int j = 0; j < b.length; j++) {
                    b[j] = Double.parseDouble(bParts[j]);
                }
            }

            return nn;
        }
    }


    // 34:03

}
