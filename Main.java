import java.io.IOException;

import network.*;
import preproccess.*;
import algebra.*;
/**
* Main
*/
public class Main {
	
	public static void main(String[] args) throws IOException {
		int dataSize = HyperParameters.DATA_SIZE;
		int testSize = HyperParameters.TEST_SIZE;
		InputImage dataImg = new InputImage();
		ConvolutionalNet convNet = new ConvolutionalNet();
		Matrix[] input = new Matrix[dataSize];
		Matrix[] tests = new Matrix[testSize];
		for(int i = 0; i < dataSize; i++){
			input[i] = convNet.forwardPropagation(dataImg.getInputs(i));
		}
		System.out.println(input[0].getRow());
		for(int i = 0; i < testSize; i++){
			tests[i] = convNet.forwardPropagation(dataImg.getTests(i));
		}
		NeuralNet network = new NeuralNet();
		InputData data = new InputData(input, tests, dataImg.getAnswers(), dataImg.getExpected());
		NetworkTrainer trainer = new NetworkTrainer(network);
		// System.out.println("Inputs:\n" + data.getInputs());
		// System.out.println("Answers:\n" + data.getAnswers());
		// System.out.println("Tests:\n" + data.getAsked());
		// System.out.println("Tests Answers:\n" + data.getExpected());
		
		// for(int i = 1; i < network.getSize(); i++)
		// 	System.out.println(network.getLayer(i));
		
		System.out.println("==============================");
		trainer.train(data);
		
		// for(int i = 1; i < network.getSize(); i++)
		// 	System.out.println(network.getLayer(i));
		
		System.out.println();
		// for(int j = 0; j < HyperParameters.TEST_SIZE; j++){
		for(int j = 0; j < testSize; j++){
			Matrix ans = network.forwardPropagation(tests[j]);
			System.out.println(ans);
			System.out.println(data.getExpected().getVector(j));
		}
	}
	
}
