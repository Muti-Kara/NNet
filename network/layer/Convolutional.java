package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class Convolutional extends Layer {
	final static double KERNEL_RANDOMIZATION = 0.001;
	int kernelSize;
	
	public Convolutional(int numOfKernels, int kernelSize) {
		parameters = new Matrix[numOfKernels+1];
		parameters[0] = new Matrix(parameters.length, 1);
		this.kernelSize = kernelSize;
		for(int i = 1; i < parameters.length; i++){
			parameters[i] = new Matrix(kernelSize, kernelSize);
		}
		for(int i = 0; i < parameters.length; i++){
			parameters[i].randomize(KERNEL_RANDOMIZATION).abs();
		}
	}
	
	public Matrix[] forwardPropagation(Matrix[] input) {
		Matrix[] ans = new Matrix[parameters.length * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 1; j < parameters.length; j++){
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
						ans[index].set(r, c, result + parameters[0].get(j, 0));
					}
				}
				ans[index].relu();
			}
		}
		return ans;
	}

	@Override
	public void applyChanges(double... learningParameters) {
		// TODO Auto-generated method stub
	}
	
}
