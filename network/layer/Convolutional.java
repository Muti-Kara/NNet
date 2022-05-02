package neuralnet.network.layer;

import neuralnet.algebra.Matrix;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional extends Layer {
	
	/**
	* Constructor takes two parameters:
	* @param number of kernels
	* @param kernels size
	 */
	public Convolutional(int type, int ... layerDescriptor) {
		super(type, layerDescriptor);
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
		// System.out.println("Input length: " + input.length);
		// System.out.println("Answer length: " + ans.length);
		for(int i = 0; i < input.length; i++){
			// System.out.println("Input " + i + ": Row: " + input[i].getRow() + ", " + input[i].getCol());
			for(int j = 0; j < information[0]; j++){
				// System.out.println("	Kernel " + j + ": Row: " + parameters[j].getRow() + ", " + parameters[j].getCol());
				int index = i * information[0] + j;
				ans[index] = input[i].convolve(parameters[j]);
				ans[index].sSum(parameters[information[0]].get(j, 0));
				ans[index].relu();
				// System.out.println("	Answer " + j + ": Row: " + ans[j].getRow() + ", " + ans[j].getCol());
			}
		}
		return ans;
	}
	
}
