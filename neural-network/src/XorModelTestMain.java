import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class XorModelTestMain {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            NeuralNetwork nn;
            try {
                nn = NeuralNetwork.load("/home/kaine/Documents/projets-perso/neural-network/xor_model.txt");
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }

            JFrame frame = new JFrame("XOR Model Test (loaded)");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout(10, 10));

            JLabel top = new JLabel("Modèle chargé: xor_model.txt  |  Teste (x,y) -> XOR(x,y)");
            frame.add(top, BorderLayout.NORTH);

            JSlider sx = slider01();
            JSlider sy = slider01();

            JPanel sliders = new JPanel(new GridLayout(2, 1, 10, 10));
            sliders.add(labeled("x", sx));
            sliders.add(labeled("y", sy));
            frame.add(sliders, BorderLayout.CENTER);

            JTextArea out = new JTextArea(10, 50);
            out.setEditable(false);
            out.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            frame.add(new JScrollPane(out), BorderLayout.SOUTH);

            JPanel right = new JPanel();
            right.setLayout(new BoxLayout(right, BoxLayout.Y_AXIS));

            JButton predictBtn = new JButton("Predict");
            JButton evalAllBtn = new JButton("Eval all 4 cases");

            JButton b00 = new JButton("Set (0,0)");
            JButton b01 = new JButton("Set (0,1)");
            JButton b10 = new JButton("Set (1,0)");
            JButton b11 = new JButton("Set (1,1)");

            right.add(predictBtn);
            right.add(Box.createVerticalStrut(8));
            right.add(evalAllBtn);
            right.add(Box.createVerticalStrut(16));
            right.add(b00);
            right.add(b01);
            right.add(b10);
            right.add(b11);

            frame.add(right, BorderLayout.EAST);

            Runnable predict = () -> {
                double x = sx.getValue() / 100.0;
                double y = sy.getValue() / 100.0;

                double[] outputs = nn.CalculateOutputs(new double[]{x, y});
                double p = outputs[0];
                int cls = (p >= 0.5) ? 1 : 0;

                out.setText(
                        "Input:  " + Arrays.toString(new double[]{x, y}) + "\n" +
                                "Output: " + Arrays.toString(outputs) + "\n" +
                                "Class (>=0.5): " + cls + "\n" +
                                "Expected (XOR rounded): " + (((x >= 0.5) ^ (y >= 0.5)) ? 1 : 0) + "\n"
                );
            };

            predictBtn.addActionListener(e -> predict.run());

            evalAllBtn.addActionListener(e -> {
                out.setText("");
                evalCase(out, nn, 0, 0);
                evalCase(out, nn, 0, 1);
                evalCase(out, nn, 1, 0);
                evalCase(out, nn, 1, 1);
            });

            b00.addActionListener(e -> { sx.setValue(0);   sy.setValue(0);   predict.run(); });
            b01.addActionListener(e -> { sx.setValue(0);   sy.setValue(100); predict.run(); });
            b10.addActionListener(e -> { sx.setValue(100); sy.setValue(0);   predict.run(); });
            b11.addActionListener(e -> { sx.setValue(100); sy.setValue(100); predict.run(); });

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            // auto show first prediction
            predict.run();
        });
    }

    static void evalCase(JTextArea out, NeuralNetwork nn, double x, double y) {
        double[] outputs = nn.CalculateOutputs(new double[]{x, y});
        double p = outputs[0];
        int cls = (p >= 0.5) ? 1 : 0;
        int expected = ((x == 1.0) ^ (y == 1.0)) ? 1 : 0;

        out.append("[" + x + ", " + y + "] -> out=" + String.format("%.6f", p)
                + " cls=" + cls + " expected=" + expected + "\n");
    }

    static JSlider slider01() {
        JSlider s = new JSlider(0, 100, 0);
        s.setMajorTickSpacing(50);
        s.setMinorTickSpacing(10);
        s.setPaintTicks(true);
        s.setPaintLabels(true);
        return s;
    }

    static JPanel labeled(String name, JSlider s) {
        JPanel p = new JPanel(new BorderLayout(8, 8));
        p.add(new JLabel(name), BorderLayout.WEST);
        p.add(s, BorderLayout.CENTER);
        return p;
    }
}
