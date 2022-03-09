package layer;

import algebra.HyperParameters;
import algebra.Matrix;
import algebra.MatrixTools;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional {
	Matrix[] kernels;
	int kernelSize;
	
	/**
	* Constructor takes two parameters:
	* @param number of kernels
	* @param kernels size
	 */
	public Convolutional(int numOfKernels, int kernelSize) {
		kernels = new Matrix[numOfKernels];
		this.kernelSize = kernelSize;
		for(int i = 0; i < kernels.length; i++){
			kernels[i] = new Matrix(kernelSize, kernelSize);
			kernels[i].randomize(HyperParameters.KERNEL_RANDOMIZATION);
		}
	}
	
	/**
	* Creates new convolutional layer from parent.
	* @param conv
	 */
	public Convolutional(Convolutional conv) {
		kernels = new Matrix[conv.kernels.length];
		this.kernelSize = conv.kernelSize;
		for(int i = 0; i < kernels.length; i++){
			kernels[i] = MatrixTools.generate(conv.getKernel(i));
		}
	}
	
	/**
	* Takes a matrix array as input. Applies its kernel with step size 1.
	* @param input
	* @return output for next layers
	 */
	public Matrix[] goForward(Matrix[] input) {
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
						ans[index].set(r, c, result);
					}
				}
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
	
	@Override
	public String toString() {
		String str = kernelSize + " " + kernels.length + "\n";
		for(int i = 0; i < kernels.length; i++){
			str += kernels[i].toString();
		}
		return str;
	}
}
