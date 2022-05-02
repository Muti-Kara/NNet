import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import dip.data.MyImage;
import dip.io.MyImageIO;
import nnet.algebra.Matrix;
import nnet.network.net.ANN;
import nnet.network.net.CNN;
import nnet.network.train.CNNTrainer;

/**
* Main
*/
public class Main {
	static String folderName = "/home/yuio/Desktop/OCR/data/";

	static Matrix[] inputs;
	static Matrix[] answers;
	
	public static void main(String[] args) throws IOException {
		
		readFolder();
		
		CNN cnn = new CNN();
		cnn.addLayer(CNN.CONVOLUTION, 2, 5);
		cnn.addLayer(CNN.MAX_POOL, 2);
		cnn.addLayer(CNN.CONVOLUTION, 2, 3);
		cnn.addLayer(CNN.MAX_POOL, 2);
		cnn.randomize(0.0001);
		
		CNNTrainer trainer = new CNNTrainer(cnn, inputs, answers, 1);
		trainer.setMiniTrainer(15000, 50, 0.001, 0.002, 0.0001);
		trainer.train(1, 1, 0, 0.001);
		
	}
	
	static public int readFolder() throws IOException{
		ArrayList<String> fileNames = new ArrayList<>();
		for(char character = 'A'; character <= 'Z'; character++){
			File[] files = new File(folderName + character).listFiles();
			for(File file : files){
				if(file.isFile())
					fileNames.add(file.getName() + " " + character);
			}
		}
		inputs = new Matrix[fileNames.size()];
		answers = new Matrix[fileNames.size()];
		for(int i = 0; i < fileNames.size(); i++){
			String file = fileNames.get(i).substring(0, fileNames.get(i).length() - 2);
			char character = fileNames.get(i).charAt(fileNames.get(i).length() - 1);
			MyImage img = MyImageIO.read(folderName + character + "/" + file);
			img = img.resize(49, 49);
			inputs[i] = new Matrix(img.convertToDouble2D()).scalarProd(0.002);
			answers[i] = new Matrix(26, 1);
			answers[i].set(character - 'A', 0, 1);
		}
		return fileNames.size();
	}

}
