package network;

import algebra.Matrix;
import algebra.MatrixTools;

/**
* A feed forward network.
* @author Muti Kara
*/
public class Network {
	CNN cnn;
	ANN ann;
	
	/**
	* Takes two arguments: 1 CNN and 1 ANN
	* @param cnn
	* @param ann
	 */
	public Network(CNN cnn, ANN ann){
		this.cnn = cnn;
		this.ann = ann;
	}
	
	/**
	 * Loads pretrained values for cnn and ann
	 * */
	public void getPreTrained(){
		// TODO: read from trained file
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
