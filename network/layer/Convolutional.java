package neuralnet.network.layer;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.*;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional extends Layer {
	int kernelSize;
	
	/**
	* Constructor takes two parameters:
	* @param number of kernels
	* @param parameters size
	 */
	public Convolutional(int numOfKernels, int kernelSize) {
		parameters = new Matrix[numOfKernels];
		bias = new Matrix(parameters.length, 1);
		this.kernelSize = kernelSize;
		for(int i = 0; i < parameters.length; i++){
			parameters[i] = new Matrix(kernelSize, kernelSize);
			parameters[i].randomize(NetworkOrganizer.kernelRandomization).abs();
		}
		bias.randomize(NetworkOrganizer.kernelRandomization).abs();
	}
	
	/**
	* Takes a matrix array as input. Applies its kernel with step size 1
	* Then applies activation function to each matrix
	* @param input
	* @return output for next layers
	 */
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[parameters.length * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < parameters.length; j++){
				int index = i * parameters.length + j;
				ans[index] = new Matrix(input[i].getRow() - kernelSize + 1, input[i].getCol() - kernelSize + 1);
				for(int r = 0; r < ans[index].getRow(); r++){
					for(int c = 0; c < ans[index].getCol(); c++){
						double result = 0;
						for(int rk = 0; rk < parameters[j].getRow(); rk++){
							for(int ck = 0; ck < parameters[j].getCol(); ck++){
								result += input[i].get(r + rk, c + ck) * parameters[j].get(rk, ck);
							}
						}
						ans[index].set(r, c, result + bias.get(j, 0));
					}
				}
				MatrixTools.relu(ans[index]);
			}
		}
		return ans;
	}
	
	/**
	* 
	* @return copy of this layer
	*/
	public Convolutional createClone() {
		Convolutional convolutional = new Convolutional(parameters.length, kernelSize);
		for(int i = 0; i < parameters.length; i++){
			convolutional.setParameter(i, parameters[i].createClone());
		}
		return convolutional;
	}
}
