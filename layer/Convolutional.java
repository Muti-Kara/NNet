package layer;

import java.io.FileWriter;
import java.io.IOException;

import algebra.HyperParameters;
import algebra.Matrix;

/**
* Convolutional Layer
* @author Muti Kara
*/
public class Convolutional {
	Matrix[] kernels;
	int kernelSize;
	
	
	public Convolutional(int numOfKernels, int kernelSize) throws IOException{
		FileWriter writer = new FileWriter(HyperParameters.projectDir + "/kernel.txt");
		kernels = new Matrix[numOfKernels];
		this.kernelSize = kernelSize;
		for(int i = 0; i < numOfKernels; i++){
			kernels[i] = new Matrix(kernelSize, kernelSize);
			kernels[i].randomize(HyperParameters.KERNEL_RANDOMIZATION);
			writer.append("\nKernel " + (i+1) + ":\n" + kernels[i].toString());
		}
		writer.close();
	}
	
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
	
	public void setKernel(int index, Matrix matrix) {
		kernels[index] = matrix;
	}
	
}
