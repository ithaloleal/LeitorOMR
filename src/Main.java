import java.io.File;
import java.util.*;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;

public class Main {

    //Define quantidade de alternativas tem as questoes
    private static final int QTD_ALTERNATIVAS = 5;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        exemplo1();
        exemplo2();
    }

    public static void exemplo1() {
        Mat image = Imgcodecs.imread("src/gabaritos/omr_test_01.png");

        // Convertendo a imagem para tons de cinza
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        // Aplicando a técnica de limiarização (threshold)
        Mat thresh = new Mat();
        Imgproc.threshold(gray, thresh, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

        // Encontrando os contornos (bordas) na imagem
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Percorrendo os contornos encontrados
        for (int i = 0; i < contours.size(); i++) {

            // Calculando a área do contorno
            //double area = Imgproc.contourArea(contours.get(i));

            //m00 do moments tambem retorna a area do contorno
            Mat circle = contours.get(i);
            Moments m = Imgproc.moments(circle);

            // Ignorando contornos muito pequenos
            if (m.m00 < 150) {
                continue;
            }

            // Encontrando os pontos extremos do contorno
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());
            double peri = Imgproc.arcLength(contour2f, true);
            Imgproc.approxPolyDP(contour2f, approxCurve, 0.025 * peri, true);
            Point[] points = approxCurve.toArray();


            // Verificando se o contorno é um círculo
            if (points.length == 8) {

                Point center = new Point((int) (m.m10 / m.m00), (int) (m.m01 / m.m00));
                int radius = (int) Math.round(Math.sqrt(m.m00 / Math.PI));

                // Extrai a região da resposta correspondente
                Rect roi = new Rect((int) center.x - radius, (int) center.y - radius, radius * 2, radius * 2);
                Mat submat = thresh.submat(roi);
                Scalar meanColor = Core.mean(submat);

                Imgproc.drawContours(image, contours, i, new Scalar(0, 0, 255), 3);
                System.out.println("Círculo encontrado na posição x=" + points[0].x + ", y=" + points[0].y);
                System.out.println("Area Total " + meanColor);

                if (meanColor.val[0] > 90){
                    Imgproc.circle(image, center, radius, new Scalar(202, 27, 27), 4, 8, 0);
                }
            }

            // Verificando se o contorno é um retângulo
            if (points.length == 4) {
                Imgproc.drawContours(image, contours, i, new Scalar(0, 255, 0), 3);
                System.out.println("Retângulo encontrado na posição x=" + points[0].x + ", y=" + points[0].y);

                //System.out.println("Total " + meanColor);
                System.out.println('\n');
            }

        }

        // Exibindo a imagem resultante
        //Imgcodecs.imwrite("C:\\Users\\PC\\Downloads\\omr_test_02.png", image);
        ImgWindow.newWindow(image);
    }

    public static void exemplo2() {
        //Mat image = Imgcodecs.imread("C:\\Users\\PC\\Downloads\\omr_test_01.png", Imgcodecs.IMREAD_GRAYSCALE);

        // Carrega a imagem do gabarito em escala cinza
        Mat gabarito = Imgcodecs.imread("src/gabaritos/omr_test_01.png", Imgcodecs.IMREAD_GRAYSCALE);

        // Aplica uma limiarização para binarizar a imagem
        //Mat binarizado = new Mat();
        //Imgproc.threshold(gabarito, binarizado, 150, 255, Imgproc.THRESH_BINARY);

        // Detecta os círculos no gabarito
        Mat circles = new Mat();
        //Imgproc.HoughCircles(gabarito, circles, Imgproc.HOUGH_GRADIENT, 1, 50, 200, 20, 10, 100);

        /*
        com os argumentos:
            cinza : Imagem de entrada (escala de cinza).
            circles : Um vetor que armazena conjuntos de 3 valores:xc,yc, rpara cada círculo detectado.
            HOUGH_GRADIENT : Defina o método de detecção. Atualmente este é o único disponível no OpenCV.
            dp = 1 : A razão inversa da resolução.
            min_dist = gray.rows/16 : Distância mínima entre os centros detectados.
            param_1 = 200 : Limiar superior para o detector interno de arestas Canny.
            param_2 = 100*: Limiar para detecção de centro.
            min_radius = 0 : Raio mínimo a ser detectado. Se desconhecido, coloque zero como padrão.
            max_radius = 0 : Raio máximo a ser detectado. Se desconhecido, coloque zero como padrão.
         */
        Imgproc.HoughCircles(gabarito, circles, Imgproc.HOUGH_GRADIENT, 1, 20, 50, 30, 10, 30);

        List<Circulo> alternativas = new ArrayList<>();
        for (int i = 0; i < circles.cols(); i++) {
            double[] circle = circles.get(0, i);
            Point center = new Point(circle[0], circle[1]);
            int radius = (int) circle[2];

            // Extrai a região de interesse correspondente à marcação
            Rect roi = new Rect((int) (center.x - radius), (int) (center.y - radius), (int) (radius * 2), (int) (radius * 2));
            Mat resposta = gabarito.submat(roi);
            //ImgWindow.newWindow(resposta);

            // Binariza a região de interesse utilizando um limiar adaptativo
            Mat respostaBinarizada = new Mat();
            Imgproc.adaptiveThreshold(resposta, respostaBinarizada, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 13, 2);
//            ImgWindow.newWindow(respostaBinarizada);

            // Calcula a área preenchida e não preenchida
            Scalar meanColor = Core.mean(respostaBinarizada);

            //System.out.println("eixo x " + roi.x + " eixo y " + roi.y + " area calculada " + meanColor.val[0] + " area " + roi.area());

            // Compara as áreas preenchida e não preenchida com um limiar previamente definido
            if (meanColor.val[0] <= 155) {
                // desenha um circulo no circulo
                /*
                img é a matriz (imagem) na qual o círculo será desenhado.
                center é o ponto central do círculo, do tipo Point.
                radius é o raio do círculo, em pixels.
                color é a cor do círculo, do tipo Scalar.
                thickness é a espessura da linha que será usada para desenhar o círculo. Um valor negativo significa que o círculo será preenchido.
                lineType é o tipo de linha usada para desenhar o círculo.
                shift é o deslocamento dos pontos de pixel.
                 */
                Imgproc.circle(gabarito, center, radius, new Scalar(0, 0, 255), 4, 8, 0);
                alternativas.add(new Circulo(roi, radius, true));
            } else {
                alternativas.add(new Circulo(roi, radius, false));
            }

        }

        //orderna primeiro pelo eixo Y para pegar questão por questao Ex: 1, 2, 3, 4
        Collections.sort(alternativas, (circulo1, circulo2) -> {
            if (circulo1.roi.y < circulo2.roi.y) {
                return -1;
            } else if (circulo1.roi.y > circulo2.roi.y) {
                return 1;
            } else {
                return 0;
            }
        });

        // agora orderna por coluna para ficar A B C D
        Collections.sort(alternativas, (circulo1, circulo2) -> {
            if (circulo1.roi.x < circulo2.roi.x && circulo1.roi.y <= circulo2.roi.y + 10) {
                return -1;
            } else if (circulo1.roi.x > circulo2.roi.x && circulo1.roi.y >= circulo2.roi.y + 10) {
                return 1;
            } else {
                return 0;
            }
        });

        //System.out.println("\n\n\n\n");

        int contQuestao = 0;
        String[] respostasMarcadas = new String[alternativas.size() / QTD_ALTERNATIVAS];
        for (int i = 0; i < alternativas.size(); i++) {
            if (alternativas.get(i).marcou) {
                if (respostasMarcadas[contQuestao] == null) {
                    int opcao = ((i + 1) % QTD_ALTERNATIVAS);
                    opcao = (opcao == 0 ? QTD_ALTERNATIVAS : opcao);
                    respostasMarcadas[contQuestao] = String.valueOf((char) (opcao + 64));
                    //System.out.println(contQuestao + 1 + ") Alternativa " + controleMarcacao[contQuestao]);
                } else {
                    //System.out.println(contQuestao + 1 + ") ANULADA!!!! ");
                    respostasMarcadas[contQuestao] = "ANULADA";
                }
            }
            if ((i + 1) % QTD_ALTERNATIVAS == 0) {
                if (i + 1 == alternativas.size() && respostasMarcadas[contQuestao] == null) {
                    respostasMarcadas[contQuestao] = "ANULADA";
                }
                contQuestao++;
            }

        }

        for (int i = 0; i < respostasMarcadas.length; i++) {
            System.out.println(i + 1 + ") " + respostasMarcadas[i]);
        }

        //System.out.println("eixo x " + circle.roi.x + " eixo y " + circle.roi.y);

        ImgWindow.newWindow(gabarito);
    }
}