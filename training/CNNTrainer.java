package training;

import network.cnn.*;
import network.ann.*;
import network.*;
import preproccess.images.*;
import preproccess.*;

import algebra.*;
import algebra.matrix.*;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer {
	Matrix[] input = new Matrix[NetworkParameters.dataSize];
	InputImage dataImg;
	CNN convNet = new CNN();
	ANN ann = new ANN();
	double best = 1e5;
	double penalty = 1e4;
	
	public CNNTrainer(InputImage img) {
		dataImg = img;
	}
	
	public void train() {
		for(int j = 0; j < NetworkParameters.cnnEpoch; j++){
			for(int i = 0; i < NetworkParameters.cnnGeneration; i++){
				System.out.println("Epoch: " + j + "\tTry: " + i + "\t\tBest: " + best);
				CNN candidate;
				if(i*j < 20)
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
		}
	}
	
	public void forwardPropagateCNN(CNN candidate) {
		for(int i = 0; i < NetworkParameters.dataSize; i++){
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
		NeuralNetwork network = new NeuralNetwork(convNet, neuralNet);
		double kernelFailures = 0;
		for(int i = 0; i < NetworkParameters.dataSize; i++){
			String str = network.classify( dataImg.getInputs(i) );
			char character = dataImg.getAnswer(i);
			System.out.println(character + str);
			if(character == str.charAt(0))
				kernelFailures -= Double.parseDouble(str.substring(str.indexOf(' ')));
			else
				kernelFailures += penalty;
		}
		return kernelFailures;
	}
	
	public CNN getBest() {
		return convNet;
	}
	
}
