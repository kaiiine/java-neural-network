import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class DrawPanel extends JPanel implements MouseListener, MouseMotionListener {
    private final int w;
    private final int h;

    private final BufferedImage canvas;
    private final Graphics2D g2;

    private int lastX = -1;
    private int lastY = -1;

    public DrawPanel(int width, int height) {
        this.w = width;
        this.h = height;
        setPreferredSize(new Dimension(w, h));
        setBackground(Color.WHITE);

        canvas = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        g2 = canvas.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setStroke(new BasicStroke(18, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g2.setColor(Color.BLACK);
        clear();

        addMouseListener(this);
        addMouseMotionListener(this);
    }

    public void clear() {
        Graphics2D gg = canvas.createGraphics();
        gg.setComposite(AlphaComposite.Src);
        gg.setColor(Color.WHITE);
        gg.fillRect(0, 0, w, h);
        gg.dispose();
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(canvas, 0, 0, null);
    }

    @Override
    public void mousePressed(MouseEvent e) {
        lastX = e.getX();
        lastY = e.getY();
        drawPoint(lastX, lastY);
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();
        if (lastX != -1) {
            g2.drawLine(lastX, lastY, x, y);
        }
        lastX = x;
        lastY = y;
        repaint();
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        lastX = -1;
        lastY = -1;
    }

    private void drawPoint(int x, int y) {
        g2.drawLine(x, y, x, y);
        repaint();
    }

    // Convertit le canvas en 28x28 -> double[784] (0..1)
    public double[] toInputVector28x28() {
        BufferedImage scaled = ImageUtils.scale(canvas, 28, 28);

        double[] v = new double[28 * 28];

        double sum = 0.0;
        double sumX = 0.0;
        double sumY = 0.0;

        int idx = 0;
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int argb = scaled.getRGB(x, y);
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = (argb) & 0xFF;

                double gray = (r + g + b) / (3.0 * 255.0);
                double ink = 1.0 - gray; // blanc -> 0, noir -> 1

                // petit seuil pour éviter le bruit
                double val = (ink < 0.05) ? 0.0 : ink;

                v[idx++] = val;

                sum += val;
                sumX += x * val;
                sumY += y * val;
            }
        }

        // Rien dessiné
        if (sum == 0.0) return v;

        // Centre de masse
        double cx = sumX / sum;
        double cy = sumY / sum;

        int dx = (int)Math.round(14.0 - cx);
        int dy = (int)Math.round(14.0 - cy);

        // Shift de l'image
        double[] shifted = new double[28 * 28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int sx = x - dx;
                int sy = y - dy;
                if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
                    shifted[y * 28 + x] = v[sy * 28 + sx];
                } else {
                    shifted[y * 28 + x] = 0.0;
                }
            }
        }

        return shifted;
    }

    // Obligatoires mais inutilisés
    @Override public void mouseMoved(MouseEvent e) {}
    @Override public void mouseClicked(MouseEvent e) {}
    @Override public void mouseEntered(MouseEvent e) {}
    @Override public void mouseExited(MouseEvent e) {}
}
