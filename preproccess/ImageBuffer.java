package preproccess;

import java.awt.image.BufferedImage;
import java.awt.Image;
import java.awt.Graphics2D;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import algebra.Matrix;
import algebra.HyperParameters;
/**
* Image
*/
public class ImageBuffer {
	BufferedImage image;
	int width, height;
	
	public ImageBuffer(String fileName) throws IOException{
		image = ImageIO.read(new File(fileName));
		width = image.getWidth();
		height = image.getHeight();
	}
	
	public void turnBlackAndWhite() {
		for(int w = 0; w < width; w++){
			for(int h = 0; h < height; h++){
				int p = image.getRGB(w, h);
				int a = p >> 24 & 0xff;
				int r = p >> 16 & 0xff;
				int g = p >> 8 & 0xff;
				int b = p & 0xff;
				int avg = a * (r + g + b)/3;
				image.setRGB(w, h, (0xff << 24) | (avg << 16) | (avg << 8) | avg);
			}
		}
	}
	
	public void maximizeContrast(){
		for(int w = 0; w < width; w++){
			for(int h = 0; h < height; h++){
				int p = image.getRGB(w, h);
				int b = p & 0xff;
				if(b < 150)
					b = 0;
				else
					b = 255;
				image.setRGB(w, h, (0xff << 24) | (b << 16) | (b << 8) | b);
			}
		}
	}
	
	public void resize(){
		width = HyperParameters.IMAGE_SIZE;
		height = HyperParameters.IMAGE_SIZE;
		Image resizedImg = image.getScaledInstance(width, height, Image.SCALE_DEFAULT);
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		Graphics2D drawer = image.createGraphics();
		drawer.drawImage(resizedImg, 0, 0, null);
		drawer.dispose();
	}
	
	public Matrix getMatrix() {
		Matrix matrix = new Matrix(width, height);
		for(int w = 0; w < width; w++){
			for(int h = 0; h < height; h++){
				int p = image.getRGB(w, h);
				int b = p & 0xff;
				matrix.set(w, h, b);
			}
		}
		return matrix;
	}
	
	public void write(String fileName) throws IOException {
		ImageIO.write(image, "png", new File(fileName));
	}
}


