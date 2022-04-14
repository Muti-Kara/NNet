package neuralnet.training;

import neuralnet.matrix.Matrix;
import neuralnet.network.Forwardable;
import neuralnet.network.NeuralNetwork;
import neuralnet.network.layer.Convolutional;
import neuralnet.network.net.ANN;
import neuralnet.network.net.CNN;
import neuralnet.network.net.Network;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer {
	Matrix[] input = new Matrix[NetworkOrganizer.dataSize];
	ImageOrganizer images;
	CNN bestConvNet = new CNN();
	CNN candidate;
	ANN ann = new ANN();
	double best = 1e7;
	double penalty = 1e3;
	
	public CNNTrainer(ImageOrganizer images) {
		this.images = images;
	}
	
	public void train() {
		for(int j = 0; j < NetworkOrganizer.cnnEpoch; j++){
			for(int i = 0; i < NetworkOrganizer.cnnGeneration; i++){
				System.out.print("\033[H\033[2J");
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
		for(int i = 0; i < NetworkOrganizer.convolutional.length; i++){
			Convolutional newLayer = ((Convolutional) (bestConvNet.getLayer(2*i))).createClone();
			if(k % NetworkOrganizer.convolutional.length == i)
				for(int j = 0; j < NetworkOrganizer.convolutional[i]; j++)
					if(k % NetworkOrganizer.convolutional[i] == j){
						newLayer.setParameter(j, (((Convolutional) (bestConvNet.getLayer(2*i))).getParameter(j)).mutate(k % NetworkOrganizer.kernel[i], NetworkOrganizer.cnnLearningRate));
						System.out.println(i + ", " + j + ", " + k % NetworkOrganizer.kernel[i]);
					}
			candidate.setLayers(2 * i, newLayer);
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
		NeuralNetwork network = new NeuralNetwork().addNet(convNet).addNet(neuralNet);;
		double kernelFailures = 0;
		for(int i = 0; i < NetworkOrganizer.dataSize; i++){
			String str = network.classify( images.getInput(i) );
			char character = images.getAnswer(i);
			System.out.println(str + " : " + character);
			if(character == str.charAt(0))
				kernelFailures -= Double.parseDouble( str.substring(str.indexOf(' ')) );
			else
				kernelFailures += penalty;
		}
		return kernelFailures / NetworkOrganizer.dataSize;
	}
	
	public CNN getBest() {
		return bestConvNet;
	}
	
}
