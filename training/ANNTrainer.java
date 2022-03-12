package training;

import algebra.NetworkParameters;
import algebra.matrix.Matrix;
import network.ann.ANN;
import network.ann.layer.FullyConnected;

/**
* NetworkTrainer
* @author Muti Kara
*/
public class ANNTrainer {
	int[] structure = NetworkParameters.structure;
	int dataSize = NetworkParameters.dataSize;
	int length = structure.length;
	
	Matrix[] activation = new Matrix[length];
	Matrix[] errors = new Matrix[length];
	FullyConnected[] changes = new FullyConnected[length];
	FullyConnected[] previousChanges = new FullyConnected[length];
	
	ANN net;
	DataOrganizer organizer;
	
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
	public void train(DataOrganizer organizer, boolean shuffle){
		this.organizer = organizer;
		int epoch = NetworkParameters.epoch;
		while(epoch-->0){
			double crossEntropy = 0;
			if(shuffle)	organizer.shuffle();
			for(int i = 0; i < NetworkParameters.stochastic; i++){
				forwardPropagate(i);
				crossEntropy += calculateError(i);
				calculateLoss();
				calculateChanges();
			}
			if(epoch % 20 == 0)
				System.out.printf("Epoch %d:\t%.8f\n", epoch, crossEntropy);
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
		activation[0] = organizer.getInput(datum);
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
		Matrix ithAnswer = organizer.getAnswer(datum);
		double crossEntropy = 0;
		for(int i = 0; i < ithAnswer.getRow(); i++){
			crossEntropy -= ithAnswer.get(i, 0) * Math.log(activation[length - 1].get(i, 0));
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
