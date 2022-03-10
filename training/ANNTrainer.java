package training;

import preproccess.InputData;
import algebra.*;
import algebra.matrix.*;
import network.ann.*;
import network.ann.layer.*;

/**
* NetworkTrainer
* @author Muti Kara
*/
public class ANNTrainer {
	int[] structure = NetworkParameters.structure;
	int dataSize = NetworkParameters.dataSize;
	int length = structure.length;
	
	Matrix[] activation = new Matrix[structure.length];
	Matrix[] errors = new Matrix[structure.length];
	FullyConnected[] changes = new FullyConnected[structure.length];
	FullyConnected[] previousChanges = new FullyConnected[structure.length];
	
	ANN net;
	Matrix input;
	Matrix answer;
	InputData data;
	boolean flag50 = true, flag20 = true, flag7 = true;
	
	/**
	* Constructor takes a neural net input.
	* @param net
	 */
	public ANNTrainer(ANN net){
		this.net = net;
		for(int i = 1; i < length; i++){
			changes[i] = new FullyConnected(net.getLayer(i));
		}
	}
	
	/**
	* takes training data as input.
	* Trains network with this data.
	* @param data
	 */
	public void train(InputData data){
		this.data = data;
		int epoch = NetworkParameters.epoch;
		while(epoch-->0){
			double crossEntropy = 0;
			for(int i = 0; i < dataSize; i++){
				forwardPropagate(i);
				crossEntropy += calculateError(i);
				calculateLoss();
				calculateChanges();
			}
			System.out.printf("Epoch %d:\t%.8f\n", epoch, crossEntropy);
			if(flag50 && crossEntropy < 50){
				NetworkParameters.learningRate *= 1.25;
				flag50 = false;
			}
			if(flag20 && crossEntropy < 20){
				NetworkParameters.learningRate *= 1.25;
				flag20 = false;
			}
			if(flag7 && crossEntropy < 7){
				NetworkParameters.learningRate *= 1.25;
				flag7 = false;
			}
			if(Double.isNaN(crossEntropy))
				return;
			applyChanges();
		}
	}
	
	/**
	* forward propagation for ith data
	* @param datum
	 */
	public void forwardPropagate(int datum) {
		activation[0] = data.getInputs()[datum];
		for(int i = 1; i < length; i++){
			activation[i] = net.getLayer(i).goForward(activation[i-1], i == length - 1);
		}
	}
	
	/**
	* 
	* @param datum
	* @return error of network in ith data
	 */
	public double calculateError(int datum) {
		Matrix ithAnswer = data.getAnswers().getVector(datum);
		double crossEntropy = 0;
		for(int i = 0; i < ithAnswer.getRow(); i++){
			crossEntropy -= ithAnswer.get(i, 0) * Math.log(activation[length - 1].get(i, 0));// + (1 - ithAnswer.get(i, 0)) * Math.log(1 - activation[length - 1].get(i, 0));
		}
		errors[length - 1] = activation[length - 1].sub(ithAnswer);
		return crossEntropy;
	}
	
	/**
	 * Calculates loss values
	 * */
	public void calculateLoss(){
		for(int j = length - 2; j > 0; j--){
			errors[j] = net.getLayer(j+1).getWeight().T().dot(errors[j+1]);
		}
	}
	
	/**
	 * Calculates changes of parameters
	 * */
	public void calculateChanges() {
		for(int j = 1; j < length; j++){
			previousChanges[j] = new FullyConnected(changes[j]);
			previousChanges[j].setWeight(changes[j].getWeight());
			previousChanges[j].setBias(changes[j].getBias());
		}
		for(int j = 1; j < length; j++){
			changes[j].setBias(changes[j].getBias().sum(errors[j]));
			changes[j].setWeight(changes[j].getWeight().sum(errors[j].dot(activation[j - 1].T())));
		}
	}
	
	/**
	 * Applies prefounded changes to parameters
	 * */
	public void applyChanges() {
		for(int i = 1; i < length; i++){
			net.getLayer(i).sub(changes[i], NetworkParameters.learningRate);
			net.getLayer(i).sub(previousChanges[i], NetworkParameters.momentumFactor);
		}
	}
}



