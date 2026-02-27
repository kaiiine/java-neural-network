import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageUtils {
    public static BufferedImage scale(BufferedImage src, int newW, int newH) {
        BufferedImage out = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, newW, newH);
        g.drawImage(src, 0, 0, newW, newH, null);
        g.dispose();
        return out;
    }
}
