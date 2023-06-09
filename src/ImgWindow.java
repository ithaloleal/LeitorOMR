/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

@SuppressWarnings("serial")
public class ImgWindow extends JPanel {
	JFrame frame;
	Graphics2D graphics;
	volatile BufferedImage img = null;
	volatile boolean clicked;
	volatile public int mouseX, mouseY;
	volatile public boolean closed;
	volatile EventQueue eventQueue = new EventQueue();

	public ImgWindow(JFrame frame) {
		this.frame = frame;
		this.frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosed (WindowEvent e) {
				closed = true;
			}
		});

		this.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked (MouseEvent e) {
				synchronized (this) {
					clicked = true;
					mouseX = e.getX();
					mouseY = e.getY();
				}
			}
		});

		this.addMouseMotionListener(new MouseMotionListener() {
			@Override
			public void mouseMoved (MouseEvent e) {
				mouseX = e.getX();
				mouseY = e.getY();
			}

			@Override
			public void mouseDragged (MouseEvent e) {
				mouseX = e.getX();
				mouseY = e.getY();
			}
		});
	}
	
//	public void processEvents() {
//		for(Runnable event: eventQueue.()) {
//			event.run();
//		}
//	}

	public void setImage (Mat mat) {
		if (mat == null) {
			img = null;
		} else {
			this.img = matToBufferedImage(mat);
			if (getWidth() != img.getWidth() || getHeight() != img.getHeight()) {
				setPreferredSize(new Dimension(img.getWidth(), img.getHeight()));
				frame.pack();
			}
		}
		repaint();
	}

	public boolean isClicked () {
		synchronized (this) {
			boolean res = clicked;
			clicked = false;
			return res;
		}
	}

	@Override
	protected void paintComponent (Graphics g) {
		super.paintComponent(g);
		BufferedImage tmp = img;
		if (tmp != null) {
			g.drawImage(tmp, 0, 0, tmp.getWidth(), tmp.getHeight(), this);
		}
	}

	public Graphics2D begin () {
		if (img != null) {
			graphics = img.createGraphics();
			return graphics;
		} else {
			return null;
		}
	}

	public void end () {
		if (graphics != null) {
			graphics.dispose();
			graphics = null;
			repaint();
		}
	}

	public void setTitle (String title) {
		frame.setTitle(title);
	}

	public void moveToDisplay (int display) {
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
		GraphicsDevice[] gd = ge.getScreenDevices();
		if (display > -1 && display < gd.length) {
			frame.setLocation(gd[display].getDefaultConfiguration().getBounds().x, frame.getY());
		} else if (gd.length > 0) {
			frame.setLocation(gd[0].getDefaultConfiguration().getBounds().x, frame.getY());
		} else {
			throw new RuntimeException("No Screens Found");
		}
	}
	
	public Mat createBuffer() {
		return new Mat(getHeight(), getWidth(), CvType.CV_8UC3);
	}
	
	public void maximize() {
		frame.setExtendedState(frame.getExtendedState() | JFrame.MAXIMIZED_BOTH);
	}
	

	public void addLabel(final String text, final Color color) {
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run () {
				JPanel panel = ImgWindow.this;
				JLabel label = new JLabel(text);
				label.setForeground(color);
				panel.add(label);		
				panel.validate();
			}
		});
	}
	
	public void clearControlls() {
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run () {
				ImgWindow.this.removeAll();
				ImgWindow.this.validate();
			}
		});
	}
	
	public static ImgWindow newUndecoratedWindow () {
		return newWindow(null, true);
	}

	public static ImgWindow newWindow () {
		return newWindow(null, false);
	}
	
	public static ImgWindow newWindow (Mat mat) {
		return newWindow(mat, false);
	}	

	public static ImgWindow newWindow (Mat mat, boolean undecorated) {
		JFrame frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setSize(400, 400);
		frame.setUndecorated(undecorated);
		ImgWindow panel = new ImgWindow(frame);
		panel.setLayout(new FlowLayout(FlowLayout.LEFT));
		frame.add(panel);
		frame.setVisible(true);
		panel.setImage(mat);
		return panel;
	}

	public static BufferedImage matToBufferedImage (Mat matrix) {		
		if (matrix.channels() == 1) {
			int cols = matrix.cols();
			int rows = matrix.rows();
			int elemSize = (int)matrix.elemSize();
			byte[] data = new byte[cols * rows * elemSize];
			int type;
			matrix.get(0, 0, data);
			switch (matrix.channels()) {
			case 1:
				type = BufferedImage.TYPE_BYTE_GRAY;
				break;
			case 3:
				type = BufferedImage.TYPE_3BYTE_BGR;
				// bgr to rgb
				byte b;
				for (int i = 0; i < data.length; i = i + 3) {
					b = data[i];
					data[i] = data[i + 2];
					data[i + 2] = b;
				}
				break;
			default:
				return null;
			}

			BufferedImage image2 = new BufferedImage(cols, rows, type);
			image2.getRaster().setDataElements(0, 0, cols, rows, data);
			return image2;
		}

		if (matrix.channels() == 3) {
			int width = matrix.width(), height = matrix.height(), channels = matrix.channels();
			byte[] sourcePixels = new byte[width * height * channels];
			matrix.get(0, 0, sourcePixels);
			// create new image and get reference to backing data
			BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
			final byte[] targetPixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
			System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);
			return image;
		}

		return null;
	}
}
