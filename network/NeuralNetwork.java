package network;

import algebra.matrix.*;
import network.ann.*;
import network.cnn.*;

/**
* A feed forward network.
* @author Muti Kara
*/
public class NeuralNetwork {
	CNN cnn;
	ANN ann;
	
	/**
	* Takes two arguments: 1 CNN and 1 ANN
	* @param cnn
	* @param ann
	 */
	public NeuralNetwork(CNN cnn, ANN ann){
		this.cnn = cnn;
		this.ann = ann;
	}
	
	/**
	* 
	* @param input
	* @return Resulting matrix after forward propagating cnn and ann
	 */
	public Matrix forwardPropagation(Matrix input){
		Matrix cnnOutputs = cnn.forwardPropagation(input);
		MatrixTools.scale(cnnOutputs);
		return ann.forwardPropagation(cnnOutputs);
	}
	
	/**
	* 
	* @param input
	* @return class of input
	 */
	public String classify(Matrix input){
		Matrix ans = forwardPropagation(input);
		int max = 0;
		for(int r = 0; r < ans.getRow(); r++)
			if(ans.get(r, 0) > ans.get(max, 0))
				max = r;
		return (char) (max + 'A') + " " + ans.get(max, 0);
	}
	
	@Override
	public String toString() {
		return cnn.toString() + "\n" + ann.toString();
	}
	
}
