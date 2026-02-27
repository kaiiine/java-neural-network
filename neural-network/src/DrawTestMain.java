import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class DrawTestMain {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // IMPORTANT :
            // - Pour des chiffres style MNIST : 784 entrées, 10 sorties
            // - Ici c'est juste un test UI : adapte la taille à ton réseau
            NeuralNetwork nn;
            try {
                nn = NeuralNetwork.load("/home/kaine/Documents/projets-perso/neural-network/digits_model.txt");
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
            JFrame frame = new JFrame("Neural Network Draw Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout());


            DrawPanel drawPanel = new DrawPanel(420, 420);
            frame.add(drawPanel, BorderLayout.CENTER);

            JTextArea outputArea = new JTextArea(6, 40);
            outputArea.setEditable(false);
            outputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            frame.add(new JScrollPane(outputArea), BorderLayout.SOUTH);

            JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT));
            JButton clearBtn = new JButton("Clear");
            JButton predictBtn = new JButton("Predict");
            buttons.add(clearBtn);
            buttons.add(predictBtn);
            frame.add(buttons, BorderLayout.WEST);

            clearBtn.addActionListener(e -> {
                drawPanel.clear();
                outputArea.setText("");
            });

            predictBtn.addActionListener(e -> {
                double[] inputs784 = drawPanel.toInputVector28x28(); // 784
                double[] outputs = nn.CalculateOutputs(inputs784);

                int predictedIndex = 0;
                for (int i = 1; i < outputs.length; i++) {
                    if (outputs[i] > outputs[predictedIndex]) predictedIndex = i;
                }

                outputArea.setText(
                        "Predicted: " + predictedIndex + "\n" +
                                "Outputs:   " + Arrays.toString(outputs) + "\n"
                );
            });

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
