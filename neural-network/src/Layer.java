import java.util.Random;

public class Layer {
    private int numNodesIn, numNodesOut;
    private double[][] costGradientW;
    private double[] costGradientB;
    private double[][] weights;
    double[] biases;

    private double[] inputs;
    private double[] activations;
    private double[] weightedInputs;

    public Layer(int numNodesIn, int numNodesOut){
        this.costGradientW = new double[numNodesIn][numNodesOut];
        this.weights = new double[numNodesIn][numNodesOut];
        this.costGradientB = new double[numNodesOut];
        this.biases=new double[numNodesOut];
        this.numNodesIn=numNodesIn;
        this.numNodesOut=numNodesOut;
    }

    // Update the weights and biases based on the cost gradients (gradient descent)
    public void ApplyGradients(double learnRate){
        for (int nodeOut=0; nodeOut<this.numNodesOut; nodeOut++){
            this.biases[nodeOut] -= this.costGradientB[nodeOut] * learnRate;
            for (int nodeIn=0; nodeIn<this.numNodesIn;nodeIn++){
                this.weights[nodeIn][nodeOut] -= this.costGradientW[nodeIn][nodeOut] * learnRate;
            }
        }
    }

    public void InitializeRandomWeights(){
        Random rng = new Random();
        for (int nodeIn=0; nodeIn<this.numNodesIn; nodeIn++){
            for (int nodeOut=0; nodeOut<this.numNodesOut; nodeOut++){
                // Get a random value between -1 and +1
                double random_value = rng.nextDouble()*2 -1;
                // Scale the random value by 1/sqrt(numInputs)
                this.weights[nodeIn][nodeOut] = random_value / Math.sqrt(this.numNodesIn);
            }
        }
    }



    public double[] caclulOutputs(double[] inputs){
        this.inputs = inputs;
        this.activations = new double[this.numNodesOut];
        this.weightedInputs = new double[this.numNodesOut];

        for(int nodeOut=0; nodeOut<this.numNodesOut; nodeOut++){
            double weightedInput = this.biases[nodeOut];
            for(int nodeIn=0; nodeIn<this.numNodesIn; nodeIn++){
                weightedInput += inputs[nodeIn] * this.weights[nodeIn][nodeOut];
            }
            weightedInputs[nodeOut] = weightedInput;
            activations[nodeOut] = activationFunction(weightedInput);
        }
        return activations;
    }

    // This function allows us to switch from a simple linear transformation to a non-linear transformation
    double activationFunction(double weightedinput){
        // return (weightedinput>0) ? 1 : 0;   // Too abrupt for the output
        return 1/(1+Math.exp(-weightedinput));   // Sigmoid function which simply smooths things out

        /*
         *  We can also choose the Hyperbolic Tangent like just below:
         * double e2w = Math.exp(2*weightedinput);
         * return (e2w-1)/(e2x+1)
         */
    }

    public double ActivationDerivative(double weightedInput){
        double activation = activationFunction(weightedInput);
        return activation * (1-activation);
    }

    public double nodeCost(double outputActivation, double expectedOutput){
        double error = outputActivation - expectedOutput;
        return error*error;
    }

    public double[] CalculateOutputLayerNodeValue(double[] expectedOutputs){
        double[] nodeValues = new double[expectedOutputs.length];
        for (int i = 0; i < nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            double costDerivative = 2 * (activations[i] - expectedOutputs[i]);
            double activationDerivative = ActivationDerivative(weightedInputs[i]);
            nodeValues[i]= activationDerivative * costDerivative;
        }
        return nodeValues;
    }

    public void UpdateGradient(double[] nodeValues){
        for (int nodeOut=0; nodeOut<this.numNodesOut; nodeOut++){
            for (int nodeIn=0; nodeIn<this.numNodesIn; nodeIn++){
                double derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];
                this.costGradientW[nodeIn][nodeOut] += derivativeCostWrtWeight;
            }
            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            this.costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }

    public double[] CalculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues){
        double[] newNodeValues = new double[this.numNodesOut];
        for (int newNodeIndex=0; newNodeIndex<newNodeValues.length; newNodeIndex++){
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex<oldNodeValues.length; oldNodeIndex++){
                double weightedInputDerivative = oldLayer.getWeights()[newNodeIndex][oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= ActivationDerivative(weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }
        return newNodeValues;
    }

    public void ClearGradients(){
        for (int i = 0; i < costGradientW.length; i++){
            for (int j = 0; j < costGradientW[i].length; j++){
                costGradientW[i][j] = 0;
            }
        }
        for (int i = 0; i < costGradientB.length; i++){
            costGradientB[i] = 0;
        }
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public int getNumNodesIn() {
        return numNodesIn;
    }

    public int getNumNodesOut() {
        return numNodesOut;
    }

    public double[] getCostGradientB() {
        return costGradientB;
    }

    public double[][] getCostGradientW() {
        return costGradientW;
    }
}
