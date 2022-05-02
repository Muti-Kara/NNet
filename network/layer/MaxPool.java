package nnet.network.layer;

import nnet.algebra.Matrix;

/**
* Pooling Layer
* @author Muti Kara
*/
public class MaxPool extends Layer {
	
	public MaxPool(int type, int ... layerDescriptor) {
		super(type, layerDescriptor);
		parameters = new Matrix[0];
	}
	
	/**
	* Applies max pooling.
	* @param input
	* @return pooled output for next layer
	 */
	@Override
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[input.length];
		for(int i = 0; i < input.length; i++){
			ans[i] = new Matrix(input[i].getRowNum() / information[0] + 1, input[i].getColNum() / information[0] + 1);
			for(int r = 0; r < ans[i].getRowNum(); r++){
				for(int c = 0; c < ans[i].getColNum(); c++){
					ans[i].set(r, c, -1e8);
				}
			}
			
			for(int r = 0; r < input[i].getRowNum(); r++){
				for(int c = 0; c < input[i].getColNum(); c++){
					ans[i].set(r/information[0], c/information[0], Math.max(input[i].get(r, c), ans[i].get(r/information[0], c/information[0])));
				}
			}
		}
		
		return ans;
	}

}
