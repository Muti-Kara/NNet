package neuralnet.network.train;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.layer.FullyConnected;
import neuralnet.network.net.ANN;

/**
* NetworkTrainer
* @author Muti Kara
*/
public class ANNTrainer {
	int[] structure = NetworkOrganizer.structure;
	int dataSize = NetworkOrganizer.dataSize;
	int length = structure.length;
	
	Matrix[] activation = new Matrix[length];
	Matrix[] errors = new Matrix[length];
	FullyConnected[] changes = new FullyConnected[length];
	FullyConnected[] prevChg = new FullyConnected[length];
	
	ANN net;
	DataOrganizer organizer;
	
	/**
	* Constructor takes a neural net input.
	* @param net
	 */
	public ANNTrainer(ANN net){
		this.net = net;
		for(int i = 1; i < length; i++){
			int action = (i == length - 1)? FullyConnected.SOFTMAX : FullyConnected.RELU;
			changes[i] = new FullyConnected(net.getLayer(i).getParameters(0).getCol(), net.getLayer(i).getParameters(0).getRow(), action);
			prevChg[i] = new FullyConnected(net.getLayer(i).getParameters(0).getCol(), net.getLayer(i).getParameters(0).getRow(), action);
			changes[i].randomize(NetworkOrganizer.annLearningRate);
		}
	}
	
	/**
	* takes training data as input.
	* Trains network with this data.
	* @param data
	 */
	public void train(DataOrganizer organizer, boolean shuffle){
		this.organizer = organizer;
		int epoch = NetworkOrganizer.epoch;
		while(epoch-->0){
			double crossEntropy = 0;
			if(shuffle)	organizer.shuffle();
			for(int i = 0; i < NetworkOrganizer.stochastic; i++){
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
			activation[i] = net.getLayer(i).forwardPropagation(new Matrix[]{ activation[i-1] })[0];
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
			errors[j] = net.getLayer(j+1).getParameters(0).T().dot(errors[j+1]);
		}
	}
	
	/**
	 * Calculates changes of parameters
	 * */
	public void calculateChanges() {
		for(int i = 1; i < length; i++) {
			prevChg[i].setParameters(0, changes[i].getParameters(0));
			prevChg[i].setParameters(1, changes[i].getParameters(1));
		}
		for(int j = 1; j < length; j++){
			changes[j].getParameters(1).sum(errors[j]);
			changes[j].getParameters(0).sum(errors[j].dot(activation[j - 1].T()));
		}
	}
	
	/**
	 * Applies prefounded changes to parameters
	 * */
	public void applyChanges() {
		for(int i = 1; i < length; i++){
			net.getLayer(i).getParameters(0).sub(changes[i].getParameters(0).sProd(NetworkOrganizer.annLearningRate));
			net.getLayer(i).getParameters(1).sub(changes[i].getParameters(1).sProd(NetworkOrganizer.annLearningRate));
			
			net.getLayer(i).getParameters(0).sub(prevChg[i].getParameters(0).sProd(NetworkOrganizer.momentumFactor));
			net.getLayer(i).getParameters(1).sub(prevChg[i].getParameters(1).sProd(NetworkOrganizer.momentumFactor));
		}
	}
}
