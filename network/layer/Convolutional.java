package neuralnet.network.layer;

import neuralnet.algebra.matrix.Matrix;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional extends Layer {
	
	/**
	* Constructor takes two parameters:
	* @param number of parameters
	* @param parameters size
	 */
	public Convolutional(int ... layerDescriptor) {
		super(layerDescriptor);
		parameters = new Matrix[information[0] + 1];
		parameters[information[0]] = new Matrix(information[0], 1);
		for(int i = 0; i < information[0]; i++){
			parameters[i] = new Matrix(information[1], information[1]);
		}
	}
	
	/**
	* Takes a matrix array as input. Applies its kernel with step size 1
	* Then applies activation function to each matrix
	* @param input
	* @return output for next layers
	 */
	@Override
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[information[0] * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < information[0]; j++){
				int index = i * information[0] + j;
				ans[index] = input[i].convolve(parameters[j]);
				ans[index].sSum(parameters[information[0]].get(j, 0));
				ans[index].relu();
			}
		}
		return ans;
	}
	
	@Override
	public Convolutional createClone() {
		Convolutional convolutional = new Convolutional(information);
		for(int i = 0; i < parameters.length; i++){
			convolutional.setParameters(i, parameters[i].createClone());
		}
		return convolutional;
	}
	
}
