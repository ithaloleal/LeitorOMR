import org.opencv.core.Rect;

class Circulo {
    public Rect roi;
    public float radius;
    public boolean marcou;

    public Circulo(Rect roi, float radius, boolean marcou) {
        this.roi = roi;
        this.radius = radius;
        this.marcou = marcou;
    }
}