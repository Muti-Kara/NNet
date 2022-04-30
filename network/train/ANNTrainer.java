package neuralnet.network.train;

import neuralnet.algebra.Matrix;
import neuralnet.network.net.ANN;

/**
* NetworkTrainer
* @author Muti Kara
*/
public class ANNTrainer extends Trainer {
	int length;
	
	Matrix[] dW, dB; // change weight, bias
	Matrix[] pW, pB; // previous changes
	Matrix[] ac, er; // activation error
	
	/**
	* Constructor takes a neural net input.
	* @param net
	 */
	public ANNTrainer(ANN net, Matrix[] inputs, Matrix[] answers, double splitRatio){
		super(net, inputs, answers, splitRatio);
		
		this.length = net.size();
		this.ac = new Matrix[length];
		this.er = new Matrix[length];
		this.dW = new Matrix[length];
		this.dB = new Matrix[length];
		this.pW = new Matrix[length];
		this.pB = new Matrix[length];
		
		for(int i = 1; i < length; i++){
			dW[i] = new Matrix(
				net.getLayer(i).getParameters(0).getRow(), 
				net.getLayer(i).getParameters(0).getCol()
			);
			dB[i] = new Matrix(
				net.getLayer(i).getParameters(1).getRow(), 
				net.getLayer(i).getParameters(1).getCol()
			);
			pW[i] = new Matrix(
				net.getLayer(i).getParameters(0).getRow(), 
				net.getLayer(i).getParameters(0).getCol()
			);
			pB[i] = new Matrix(
				net.getLayer(i).getParameters(1).getRow(), 
				net.getLayer(i).getParameters(1).getCol()
			);
		}
	}
	
	@Override
	public void stochastic(int epoch, int stochastic) {
		Matrix[] data = nextData();
		forwardPropagate(data[0]);
		calculateError(data[1]);
		calculateLoss();
		calculateChanges();
	}

	/**
	 * Applies prefounded changes to parameters
	 * */
	@Override
	public void apply(double rate, double momentum) {
		System.out.printf("%.8f\n", error);
		
		for(int i = 1; i < length; i++){
			net.getLayer(i).getParameters(0).sub(dW[i].sProd(rate));
			net.getLayer(i).getParameters(1).sub(dB[i].sProd(rate));
			
			net.getLayer(i).getParameters(0).sub(pW[i].sProd(momentum));
			net.getLayer(i).getParameters(1).sub(pB[i].sProd(momentum));
		}
		for(int i = 1; i < length; i++) {
			pW[i] = dW[i];
			pB[i] = dB[i];
			
			dW[i].sProd(0);
			dB[i].sProd(0);
		}
	}

	/**
	* 
	* @param datum
	* @return error of network in ith data
	 */
	@Override
	public void calculateError(Matrix answer) {
		double crossEntropy = 0;
		for(int i = 0; i < answer.getRow(); i++){
			crossEntropy -= answer.get(i, 0) * Math.log(ac[length - 1].get(i, 0));
		}
		er[length - 1] = ac[length - 1].sub(answer);
		error += crossEntropy;
	}
	
	/**
	* forward propagation for ith data
	* @param datum
	 */
	public void forwardPropagate(Matrix input) {
		ac[0] = input;
		for(int i = 1; i < length; i++){
			ac[i] = net.getLayer(i).forwardPropagation(new Matrix[]{ ac[i-1] })[0];
		}
	}
	
	/**
	 * Calculates loss values
	 * */
	public void calculateLoss(){
		for(int i = length - 2; i > 0; i--){
			er[i] = net.getLayer(i+1).getParameters(0).T().dot(er[i+1]);
		}
	}
	
	/**
	 * Calculates changes of parameters
	 * */
	public void calculateChanges() {
		for(int i = 1; i < length; i++){
			dB[i].sum(er[i]);
			dW[i].sum(er[i].dot(ac[i - 1].T()));
		}
	}

}
