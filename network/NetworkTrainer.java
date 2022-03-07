package network;

import layer.FullyConnected;
import preproccess.InputData;
import algebra.*;

/**
* NetworkTrainer
*/
public class NetworkTrainer {
	int[] structure = HyperParameters.structure;
	int dataSize = HyperParameters.DATA_SIZE;
	int length = structure.length;
	
	Matrix[] activation = new Matrix[structure.length];
	Matrix[] errors = new Matrix[structure.length];
	FullyConnected[] changes = new FullyConnected[structure.length];
	
	NeuralNet net;
	Matrix input;
	Matrix answer;
	InputData data;
	
	public NetworkTrainer(NeuralNet net){
		this.net = net;
		for(int i = 1; i < length; i++){
			changes[i] = new FullyConnected(net.getLayer(i));
		}
	}
	
	public void train(InputData data){
		this.data = data;
		int epoch = HyperParameters.EPOCH;
		while(epoch-->0){
			double crossEntropy = 0;
			for(int i = 0; i < dataSize; i++){
				forwardPropagate(i);
				crossEntropy += calculateError(i);
				calculateLoss();
				calculateChanges();
			}
			// if(epoch < 1)
			System.out.printf("Epoch %d:\t%.8f\n", epoch, crossEntropy);
			if(Double.isNaN(crossEntropy))
				return;
			applyChanges();
		}
	}
	
	public void forwardPropagate(int i) {
		activation[0] = data.getInputs()[i];
		for(int j = 1; j < length; j++){
			activation[j] = net.layers[j].goForward(activation[j-1], j == length - 1);
		}
	}
	
	public double calculateError(int i) {
		Matrix ithAnswer = data.getAnswers().getVector(i);
		double crossEntropy = 0;
		for(int j = 0; j < ithAnswer.getRow(); j++){
			crossEntropy -= ithAnswer.get(j, 0) * Math.log(activation[length - 1].get(j, 0));
			// System.out.println(activation[length - 1].get(j, 0));
		}
		errors[length - 1] = activation[length - 1].sub(ithAnswer);
		return crossEntropy;
	}
	
	public void calculateLoss(){
		for(int j = length - 2; j > 0; j--){
			errors[j] = net.getLayer(j+1).getWeight().T().dot(errors[j+1]);
		}
	}
	
	public void calculateChanges() {
		for(int j = 1; j < length; j++){
			changes[j].setBias(changes[j].getBias().sum(errors[j]));
			changes[j].setWeight(changes[j].getWeight().sum(errors[j].dot(activation[j - 1].T())));
		}
	}
	
	public void applyChanges() {
		for(int i = 1; i < length; i++){
			net.layers[i].sub(changes[i]);
		}
	}
}



