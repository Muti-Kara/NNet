package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class Convolutional extends Layer {
	Matrix[] activation, error;
	int kernelSize;
	
	public Convolutional(int ... layerDescription) {
		this.training = layerDescription[layerDescription.length - 1] == TRAINMENT;
		
		parameters = new Matrix[layerDescription[0]+1];
		parameters[0] = new Matrix(parameters.length, 1);
		this.kernelSize = layerDescription[1];
		for(int i = 1; i < parameters.length; i++){
			parameters[i] = new Matrix(kernelSize, kernelSize);
		}
		
		if (this.training) {
			deltaPara = new Matrix[parameters.length];
			prevDelta = new Matrix[parameters.length];
			for(int i = 0; i < parameters.length; i++){
				deltaPara[i] = new Matrix(parameters[i].getRow(), parameters[i].getCol());
				prevDelta[i] = new Matrix(parameters[i].getRow(), parameters[i].getCol());
			}
		}
	}
	
	@Override
	public Matrix[] forwardPropagation(Object inputs) {
		Matrix[] input;
		if (inputs instanceof Matrix[])
			input = (Matrix[]) inputs;
		else
			return null;
		
		if(training)
			activation = input;
		
		Matrix[] result = new Matrix[(parameters.length - 1) * input.length];
		for(int i = 0; i < input.length; i++){
			for(int j = 1; j < parameters.length; j++){
				int index = i * (parameters.length - 1) + j - 1;
				result[index] = input[i].convolve(parameters[j], parameters[j].getRow() / 2);
				result[index].scalarSum(parameters[0].get(j, 0));
				result[index].relu();
			}
		}
		
		return result;
	}
	
	@Override
	public Matrix[] backPropagation(Object errors) {
		Matrix[] error;
		if (errors instanceof Matrix[])
			error = (Matrix[]) errors;
		else
			return null;
		
		Matrix[] newError = new Matrix[activation.length];
		
		this.error = new Matrix[error.length];
		for (int i = 0; i < error.length; i++) {
			this.error[i] = error[i].d_relu();
		}
		
		for (int i = 0; i < activation.length; i++) {
			newError[i] = new Matrix(activation[i].getRow(), activation[i].getCol());
			for (int j = 1; j < parameters.length; j++) {
				int index = i * (parameters.length - 1) + j - 1;
				newError[i].sum( this.error[ index ].convolve(parameters[j], parameters[j].getRow() / 2) );
			}
		}
		
		return newError;
	}
	
	public void calculateChanges() {
		for (int i = 0; i < activation.length; i++) {
			for (int j = 1; j < parameters.length; j++) {
				int index = i * (parameters.length - 1) + j - 1;
				deltaPara[j].sum( this.error[ index ].convolve(activation[i], parameters[j].getRow() / 2));
				deltaPara[0].set(j, 0, deltaPara[0].get(j, 0) + this.error[ index ].totalSum());
			}
		}
	}
	
}
