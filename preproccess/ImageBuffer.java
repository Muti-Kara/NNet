package preproccess;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

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
				int r = p >> 16 & 0xff;
				int g = p >> 8 & 0xff;
				int b = p & 0xff;
				int avg = (r + g + b)/3;
				image.setRGB(w, h, (0xff << 24) | (avg << 16) | (avg << 8) | avg);
			}
		}
	}
	
	public void poolMax(int size) throws IOException{
		BufferedImage newImage = new BufferedImage(width / size, height / size, BufferedImage.TYPE_INT_ARGB);
		for(int w = 0; w < width - size; w += size){
			for(int h = 0; h < height - size; h += size){
				int MAX = 0;
				for(int ww = 0; ww < size; ww++){
					for(int hh = 0; hh < size; hh++){
						MAX = Math.max(MAX, image.getRGB(w + ww, h + hh) >> 16 & 0xff);
					}
				}
				newImage.setRGB(w/size, h/size, (0xff << 24) | (MAX << 16) | (MAX << 8) | MAX);
			}
		}
		width = width / size;
		height = height / size;
		image = newImage;
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
	
	public void write(String fileName) throws IOException {
		ImageIO.write(image, "png", new File(fileName));
	}
}


