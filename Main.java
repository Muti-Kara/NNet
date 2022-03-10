import java.io.IOException;

import algebra.NetworkParameters;
import network.NeuralNetwork;
import preproccess.images.InputImage;
import preproccess.parameters.Reader;

/**
* Main
*/
public class Main {
	
	public static void main(String[] args) throws IOException {
		
		NetworkParameters.addConvolutionalLayer(2, 5, 2);
		NetworkParameters.addConvolutionalLayer(2, 3, 2);
		NetworkParameters.addFullyConnectedLayer(256);
		NetworkParameters.addFullyConnectedLayer(30);
		NetworkParameters.addFullyConnectedLayer(26);
		
		NetworkParameters.setImageSize(37);
		
		InputImage dataImg = new InputImage();
		NeuralNetwork network = Reader.readNN();
		for(int index = 0; index < NetworkParameters.testSize; index++){
			String str = network.classify( dataImg.getTests(index) );
			System.out.println(dataImg.getExpected(index) + str);
		}
		
	}
	
}




