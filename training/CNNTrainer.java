package training;

import network.*;
import preproccess.*;
import algebra.*;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer {
	Matrix[] input = new Matrix[HyperParameters.DATA_SIZE];
	InputImage dataImg;
	CNN convNet = new CNN();
	ANN ann = new ANN();
	double best = 1e5;
	
	public CNNTrainer(InputImage img) {
		dataImg = img;
	}
	
	public void train() {
		for(int j = 0; j < HyperParameters.CNN_EPOCH; j++){
			for(int i = 0; i < HyperParameters.CNN_GENERATION; i++){
				CNN candidate;
				if(i*j < 5)
					candidate = new CNN();
				else
					candidate = new CNN(convNet);
				forwardPropagateCNN(candidate);
				ANN candidateTester = miniTrainmentANN();
				double candidateError = calculateKernelErrors(candidate, candidateTester);
				if(!Double.isNaN(candidateError) && candidateError < best){
					convNet = candidate;
					ann = candidateTester;
					best = candidateError;
				}
			}
			Network network = new Network(convNet, ann);
			double kernelFailures = HyperParameters.TEST_SIZE;
			for(int i = 0; i < HyperParameters.TEST_SIZE; i++){
				String str = network.classify( dataImg.getTests(i) );
				System.out.println((char) ('A' + i) + str);
				if('A' + i == str.charAt(0))
					kernelFailures -= Double.parseDouble(str.substring(str.indexOf(' ')));
				else
					kernelFailures += 3;
			}
			System.out.println(kernelFailures);
		}
	}
	
	public void forwardPropagateCNN(CNN candidate) {
		for(int i = 0; i < HyperParameters.DATA_SIZE; i++){
			input[i] = MatrixTools.scale(candidate.forwardPropagation(dataImg.getInputs(i)));
		}
	}
	
	public ANN miniTrainmentANN() {
		ANN ann = new ANN();
		InputData data = new InputData(input, dataImg.getAnswers());
		ANNTrainer trainer = new ANNTrainer(ann);
		trainer.train(data);
		return ann;
	}
	
	public double calculateKernelErrors(CNN convNet, ANN neuralNet) {
		Network network = new Network(convNet, neuralNet);
		double kernelFailures = HyperParameters.TEST_SIZE;
		for(int i = 0; i < HyperParameters.TEST_SIZE; i++){
			String str = network.classify( dataImg.getInputs(9*i) );
			if('A' + i == str.charAt(0))
				kernelFailures -= Double.parseDouble(str.substring(str.indexOf(' ')));
			else
				kernelFailures += 3;
		}
		return kernelFailures;
	}
	
}
