package preproccess.images;

import java.awt.image.BufferedImage;
import java.awt.Image;
import java.awt.Graphics2D;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import algebra.matrix.Matrix;
import algebra.NetworkParameters;

/**
* Image
* @author Muti Kara
*/
public class ImageBuffer {
	BufferedImage image;
	int width, height;
	
	/**
	* Opens given image file
	* @param fileName
	* @throws IOException
	 */
	public ImageBuffer(String fileName) throws IOException{
		image = ImageIO.read(new File(fileName));
		width = image.getWidth();
		height = image.getHeight();
	}
	
	/**
	 * Converts image to black and white.
	 * */
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
	
	/**
	 * Makes picture consist of black and white pixels.
	 * */
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
	
	/**
	 * Resizes image
	 * */
	public void resize(){
		width = NetworkParameters.imageSize;
		height = NetworkParameters.imageSize;
		Image resizedImg = image.getScaledInstance(width, height, Image.SCALE_DEFAULT);
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		Graphics2D drawer = image.createGraphics();
		drawer.drawImage(resizedImg, 0, 0, null);
		drawer.dispose();
	}
	
	/**
	* Converts image to a matrix to use in neural network.
	* @return output matrix
	 */
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
	
	/**
	* Writes image to given file. This stands for debugging purposes.
	* @param fileName
	* @throws IOException
	 */
	public void write(String fileName) throws IOException {
		ImageIO.write(image, "png", new File(fileName));
	}
}


