package neuralnet.network.cnn.layer;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.*;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional {
	Matrix[] kernels;
	Matrix kernelBiases;
	int kernelSize;
	
	/**
	* Constructor takes two parameters:
	* @param number of kernels
	* @param kernels size
	 */
	public Convolutional(int numOfKernels, int kernelSize) {
		kernels = new Matrix[numOfKernels];
		kernelBiases = new Matrix(kernels.length, 1);
		this.kernelSize = kernelSize;
		for(int i = 0; i < kernels.length; i++){
			kernels[i] = new Matrix(kernelSize, kernelSize);
			kernels[i].randomize(NetworkOrganizer.kernelRandomization).abs();
		}
		kernelBiases.randomize(NetworkOrganizer.kernelRandomization).abs();
	}
	
	/**
	* Takes a matrix array as input. Applies its kernel with step size 1
	* Then applies activation function to each matrix
	* @param input
	* @return output for next layers
	 */
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[kernels.length * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < kernels.length; j++){
				int index = i * kernels.length + j;
				ans[index] = new Matrix(input[i].getRow() - kernelSize + 1, input[i].getCol() - kernelSize + 1);
				for(int r = 0; r < ans[index].getRow(); r++){
					for(int c = 0; c < ans[index].getCol(); c++){
						double result = 0;
						for(int rk = 0; rk < kernels[j].getRow(); rk++){
							for(int ck = 0; ck < kernels[j].getCol(); ck++){
								result += input[i].get(r + rk, c + ck) * kernels[j].get(rk, ck);
							}
						}
						ans[index].set(r, c, result + kernelBiases.get(j, 0));
					}
				}
				MatrixTools.relu(ans[index]);
			}
		}
		return ans;
	}
	
	/**
	 * 
	 * @param index
	 * @return index th kernel
	 */
	public Matrix getKernel(int index) {
		return kernels[index];
	}
	
	public void setKernel(int index, Matrix kernel) {
		kernels[index] = kernel;
	}
	
	public Matrix getKernelBiases() {
		return kernelBiases;
	}
	
	public void setKernelBiases(Matrix kernelBiases) {
		this.kernelBiases = kernelBiases;
	}
	
	public Convolutional createClone() {
		Convolutional convolutional = new Convolutional(kernels.length, kernelSize);
		for(int i = 0; i < kernels.length; i++){
			convolutional.setKernel(i, kernels[i].createClone());
		}
		return convolutional;
	}
	
	@Override
	public String toString() {
		String str = kernelSize + " " + kernels.length + "\n";
		str += kernelBiases.toString() + "\n\n";
		for(int i = 0; i < kernels.length; i++){
			str += kernels[i].toString() + "\n";
		}
		return str;
	}
}
