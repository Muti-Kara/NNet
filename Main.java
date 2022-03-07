import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import network.*;
import preproccess.*;
import algebra.*;

/**
* Main
*/
public class Main {
	
	public static void main(String[] args) throws IOException {
		ConvolutionalNet convNet = new ConvolutionalNet();
		NeuralNet neuralNet = new NeuralNet();
		
		InputImage dataImg = new InputImage();
		Matrix[] input = new Matrix[HyperParameters.DATA_SIZE];
		for(int i = 0; i < HyperParameters.DATA_SIZE; i++){
			input[i] = MatrixTools.scale(convNet.forwardPropagation(dataImg.getInputs(i)));
		}
		
		InputData data = new InputData(input, dataImg.getAnswers());
		NetworkTrainer trainer = new NetworkTrainer(neuralNet);
		trainer.train(data);
		
		Network network = new Network(convNet, neuralNet);
		for(int i = 0; i < HyperParameters.TEST_SIZE; i++){
			Matrix ans = network.forwardPropagation( dataImg.getTests(i) );
			System.out.println(ans);
		}
		
		FileWriter writer = new FileWriter(new File(HyperParameters.projectDir + "/NET.txt"));
		writer.write(network.toString());
		writer.close();
	}
	
}
