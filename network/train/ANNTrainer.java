package nnet.network.train;

import nnet.algebra.Matrix;
import nnet.network.net.ANN;

/**
* NetworkTrainer
* Uses backpropagation algorithm to train ANN's
* @author Muti Kara
*/
public class ANNTrainer extends Trainer {
	int length;
	double error = 0;
	
	Matrix[] dW, dB; // change weight, bias
	Matrix[] pW, pB; // previous changes
	Matrix[] ac, er; // activation error
	
	/**
	* Creates change (dW, dB), previous change (pW, pB), activation (ac), and error (er) matrices.
	* @param net
	* @param inputs
	* @param answers
	* @param splitRatio
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
				net.getLayer(i).getParameter(0).getRowNum(), 
				net.getLayer(i).getParameter(0).getColNum()
			);
			dB[i] = new Matrix(
				net.getLayer(i).getParameter(1).getRowNum(), 
				net.getLayer(i).getParameter(1).getColNum()
			);
			pW[i] = new Matrix(
				net.getLayer(i).getParameter(0).getRowNum(), 
				net.getLayer(i).getParameter(0).getColNum()
			);
			pB[i] = new Matrix(
				net.getLayer(i).getParameter(1).getRowNum(), 
				net.getLayer(i).getParameter(1).getColNum()
			);
		}
	}

	@Override
	public void preStochastic(int atEpoch, int stochastic, double momentum) {
		error = 0;
	}
	
	/**
	 * classic backpropagation algorithm
	 * */
	@Override
	public void stochastic(int at) {
		Matrix[] data = nextData();
		forwardPropagation(data[0]);
		calculateError(data[1]);
		calculateLoss();
		calculateChanges();
	}

	/**
	 * Applies prefounded changes to parameters
	 * */
	@Override
	public void postStochastic(double rate, double momentum) {
		if(Double.isNaN(error)){
			System.out.println("Invalid error value");
			return;
		}
		
		System.out.printf("%.8f\n", error);
		
		for(int i = 1; i < length; i++){
			net.getLayer(i).getParameter(0).sub(dW[i].scalarProd(rate));
			net.getLayer(i).getParameter(1).sub(dB[i].scalarProd(rate));
			
			net.getLayer(i).getParameter(0).sub(pW[i].scalarProd(momentum/rate));
			net.getLayer(i).getParameter(1).sub(pB[i].scalarProd(momentum/rate));
		}
		
		for(int i = 1; i < length; i++) {
			pW[i] = dW[i];
			pB[i] = dB[i];
			
			dW[i].scalarProd(0);
			dB[i].scalarProd(0);
		}
	}

	/**
	* 
	* @param datum
	* @return error of network according to given asnwer matrix
	 */
	public void calculateError(Matrix answer) {
		double crossEntropy = 0;
		for(int i = 0; i < answer.getRowNum(); i++){
			crossEntropy -= answer.get(i, 0) * Math.log(ac[length - 1].get(i, 0));
		}
		er[length - 1] = ac[length - 1].sub(answer);
		error += crossEntropy;
	}
	
	/**
	* forward propagation for given input matrix
	* @param datum
	 */
	public void forwardPropagation(Matrix input) {
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
			er[i] = net.getLayer(i+1).getParameter(0).transpose().dot(er[i+1]);
		}
	}
	
	/**
	 * Calculates changes of parameters
	 * */
	public void calculateChanges() {
		for(int i = 1; i < length; i++){
			dB[i].sum(er[i]);
			dW[i].sum(er[i].dot(ac[i - 1].transpose()));
		}
	}
	
	public double getError() {
		return error;
	}

}
