package neuralnet.network.layer;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional extends Layer {
	int numOfKernels, kernelSize;
	
	/**
	* Constructor takes two parameters:
	* @param number of parameters
	* @param parameters size
	 */
	public Convolutional(int numOfKernels, int kernelSize) {
		this.kernelSize = kernelSize;
		this.numOfKernels = numOfKernels;
		
		parameters = new Matrix[numOfKernels + 1];
		parameters[numOfKernels] = new Matrix(numOfKernels, 1);
		for(int i = 0; i < numOfKernels; i++){
			parameters[i] = new Matrix(kernelSize, kernelSize);
			parameters[i].randomize(NetworkOrganizer.kernelRandomization).abs();
		}
		parameters[numOfKernels].randomize(NetworkOrganizer.kernelRandomization).abs();
	}
	
	/**
	* Takes a matrix array as input. Applies its kernel with step size 1
	* Then applies activation function to each matrix
	* @param input
	* @return output for next layers
	 */
	@Override
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[numOfKernels * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < numOfKernels; j++){
				int index = i * numOfKernels + j;
				ans[index] = new Matrix(input[i].getRow() - kernelSize + 1, input[i].getCol() - kernelSize + 1);
				for(int r = 0; r < ans[index].getRow(); r++){
					for(int c = 0; c < ans[index].getCol(); c++){
						double result = 0;
						for(int rk = 0; rk < parameters[j].getRow(); rk++){
							for(int ck = 0; ck < parameters[j].getCol(); ck++){
								result += input[i].get(r + rk, c + ck) * parameters[j].get(rk, ck);
							}
						}
						ans[index].set(r, c, result + parameters[numOfKernels].get(j, 0));
					}
				}
				ans[index].relu();
			}
		}
		return ans;
	}
	
	public Convolutional createClone() {
		Convolutional convolutional = new Convolutional(numOfKernels, kernelSize);
		for(int i = 0; i < numOfKernels; i++){
			convolutional.setParameters(i, parameters[i].createClone());
		}
		return convolutional;
	}
	
	@Override
	public String toString() {
		String str = kernelSize + " " + numOfKernels + "\n";
		str += parameters[numOfKernels].toString() + "\n\n";
		for(int i = 0; i < numOfKernels; i++){
			str += parameters[i].toString() + "\n";
		}
		return str;
	}
}
