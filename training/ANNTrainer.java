package neuralnet.training;

import neuralnet.NetworkOrganizer;
import neuralnet.matrix.Matrix;
import neuralnet.network.layer.FullyConnected;
import neuralnet.network.net.ANN;

/**
* NetworkTrainer
* @author Muti Kara
*/
public class ANNTrainer {
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
		for(int i = 1; i < length-1; i++){
			changes[i] = (FullyConnected) new FullyConnected(net.getLayer(i).getParameter(0).getCol(), net.getLayer(i).getParameter(0).getRow(), FullyConnected.RELU_ACTIVATION).randomize(NetworkOrganizer.annLearningRate);
			previousChanges[i] = new FullyConnected(net.getLayer(i).getParameter(0).getCol(), net.getLayer(i).getParameter(0).getRow(), FullyConnected.RELU_ACTIVATION);
		}
		changes[length-1] = (FullyConnected) new FullyConnected(net.getLayer(length-1).getParameter(0).getCol(), net.getLayer(length-1).getParameter(0).getRow(), FullyConnected.SOFTMAX_ACTIVATION).randomize(NetworkOrganizer.annLearningRate);
		previousChanges[length-1] = new FullyConnected(net.getLayer(length-1).getParameter(0).getCol(), net.getLayer(length-1).getParameter(0).getRow(), FullyConnected.SOFTMAX_ACTIVATION);
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
			activation[i] = net.getLayer(i).forwardPropagation(new Matrix[] { activation[i-1] })[0];
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
			errors[j] = net.getLayer(j+1).getParameter(0).transpose().dot(errors[j+1]);
		}
	}
	
	/**
	 * Calculates changes of parameters
	 * */
	public void calculateChanges() {
		for(int i = 1; i < length; i++) {
			previousChanges[i].setParameter(0, changes[i].getParameter(0));
			previousChanges[i].setBias(changes[i].getBias());
		}
		for(int j = 1; j < length; j++){
			changes[j].getBias().sum(errors[j]);
			changes[j].getParameter(0).sum(errors[j].dot(activation[j - 1].transpose()));
		}
	}
	
	/**
	 * Applies prefounded changes to parameters
	 * */
	public void applyChanges() {
		for(int i = 1; i < length; i++){
			net.getLayer(i).getParameter(0).sub(changes[i].getParameter(0).scalarProd(NetworkOrganizer.annLearningRate));
			net.getLayer(i).getBias().sub(changes[i].getBias().scalarProd(NetworkOrganizer.annLearningRate));
			
			net.getLayer(i).getParameter(0).sub(previousChanges[i].getParameter(0).scalarProd(NetworkOrganizer.momentumFactor));
			net.getLayer(i).getBias().sub(previousChanges[i].getBias().scalarProd(NetworkOrganizer.momentumFactor));
		}
	}
}
