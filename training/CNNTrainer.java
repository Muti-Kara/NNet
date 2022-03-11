package training;

import network.cnn.*;
import network.ann.*;
import network.*;
import preproccess.images.*;

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
	CNN candidate;
	ANN ann = new ANN();
	double best = 1e7;
	double penalty = 1e3;
	
	public CNNTrainer(InputImage img) {
		dataImg = img;
	}
	
	public void train() {
		for(int j = 0; j < NetworkParameters.cnnEpoch; j++){
			for(int i = 0; i < NetworkParameters.cnnGeneration; i++){
				System.out.println("Epoch: " + j + "\tTry: " + i + "\t\tBest: " + best);
				if(i*j < 20)
					candidate = new CNN();
				else
					candidate = new CNN(convNet);
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
	
	public ANN miniTrainmentANN() {
		ANN ann = new ANN();
		DataOrganizer organizer = new DataOrganizer(dataImg);
		ANNTrainer trainer = new ANNTrainer(ann);
		organizer.convert(candidate);
		trainer.train(organizer);
		return ann;
	}
	
	public double calculateKernelErrors(CNN convNet, ANN neuralNet) {
		NeuralNetwork network = new NeuralNetwork(convNet, neuralNet);
		double kernelFailures = 0;
		for(int i = 0; i < NetworkParameters.dataSize; i++){
			String str = network.classify( dataImg.getInputs(i) );
			char character = dataImg.getAnswer(i);
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
