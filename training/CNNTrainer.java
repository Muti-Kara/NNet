package training;

import network.cnn.*;
import network.cnn.layer.Convolutional;
import network.ann.*;
import network.*;
import preproccess.images.*;

import java.util.Random;

import algebra.*;
import algebra.matrix.*;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer {
	Matrix[] input = new Matrix[NetworkParameters.dataSize];
	Random rand = new Random();
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
				generateCandidate(i);
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
	
	public void generateCandidate(int k) {
		candidate = new CNN();
		int randomInt = rand.nextInt();
		for(int i = 0; i < NetworkParameters.convolutional.length; i++){
			Convolutional newLayer = bestConvNet.getConvLayer(i).createClone();
			if(randomInt % NetworkParameters.convolutional.length == i)
				for(int j = 0; j < NetworkParameters.convolutional[i]; j++)
					if(randomInt % NetworkParameters.convolutional[i] == j)
						newLayer.setKernel(j, MatrixTools.generate(bestConvNet.getConvLayer(i).getKernel(j), randomInt % NetworkParameters.kernel[i]));;
			candidate.setConvLayer(i, newLayer);
		}
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
		return kernelFailures / NetworkParameters.dataSize;
	}
	
	public CNN getBest() {
		return bestConvNet;
	}
	
}
