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
	InputImage images;
	CNN bestConvNet = new CNN();
	CNN candidate;
	ANN ann = new ANN();
	double best = 1e7;
	double penalty = 1e3;
	
	public CNNTrainer(InputImage images) {
		this.images = images;
	}
	
	public void train() {
		for(int j = 0; j < NetworkParameters.cnnEpoch; j++){
			for(int i = 0; i < NetworkParameters.cnnGeneration; i++){
				System.out.println("Epoch: " + j + "\tTry: " + i + "\t\tBest: " + best);
				generateCandidate();
				ANN candidateTester = miniTrainmentANN();
				double candidateError = calculateKernelErrors(candidate, candidateTester);
				if(!Double.isNaN(candidateError) && candidateError < best){
					bestConvNet = candidate;
					ann = candidateTester;
					best = candidateError;
				}
			}
		}
	}
	
	public void generateCandidate() {
		// TODO: write a method to generate candidates more logically
		candidate = new CNN();
	}
	
	public ANN miniTrainmentANN() {
		ANN ann = new ANN();
		DataOrganizer organizer = new DataOrganizer(images);
		ANNTrainer trainer = new ANNTrainer(ann);
		organizer.convert(candidate);
		trainer.train(organizer, false);
		return ann;
	}
	
	public double calculateKernelErrors(CNN convNet, ANN neuralNet) {
		NeuralNetwork network = new NeuralNetwork(convNet, neuralNet);
		double kernelFailures = 0;
		for(int i = 0; i < NetworkParameters.dataSize; i++){
			String str = network.classify( images.getInput(i) );
			char character = images.getAnswer(i);
			if(character == str.charAt(0))
				kernelFailures -= Double.parseDouble( str.substring(str.indexOf(' ')) );
			else
				kernelFailures += penalty;
		}
		return kernelFailures;
	}
	
	public CNN getBest() {
		return bestConvNet;
	}
	
}
