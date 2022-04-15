package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class FullyConnected extends Layer {
	public final static int IDENTITY_ACTIVATION = 0;
	public final static int RELU_ACTIVATION = 1;
	public final static int SOFTMAX_ACTIVATION = 2;
	
	Matrix activation, error;
	int activationType;
	
	public FullyConnected(int previous, int next, int activationType, int training){
		this.training = training == TRAINMENT;
		
		this.activationType = activationType;
		
		parameters = new Matrix[2];
		parameters[0] = new Matrix(next, previous);
		parameters[1] = new Matrix(next, 1);
		
		if (this.training) {
			deltaPara = new Matrix[2];
			deltaPara[0] = new Matrix(next, previous);
			deltaPara[1] = new Matrix(next, 1);
			
			prevDelta = new Matrix[2];
			prevDelta[0] = new Matrix(next, previous);
			prevDelta[1] = new Matrix(next, 1);
		}
	}
	
	@Override
	public Matrix forwardPropagation(Object inputs){
		Matrix input;
		if(inputs instanceof Matrix)
			input = (Matrix) inputs;
		else
			return null;
		
		if(training)
			activation = input;
		
		switch (activationType) {
			case IDENTITY_ACTIVATION:
				return input;
			
			case RELU_ACTIVATION:
				return parameters[0].dot(input).sum(parameters[1]).relu();
			
			case SOFTMAX_ACTIVATION:
				return parameters[0].dot(input).sum(parameters[1]).softmax();
			
			default:
				System.out.println("Activation function is undefined!");
				return null;
		}
	}
	
	@Override
	public Matrix backPropagation(Object errors) {
		Matrix error;
		if(errors instanceof Matrix)
			error = (Matrix) errors;
		else
			return null;
		
		switch (activationType) {
			case IDENTITY_ACTIVATION:
				this.error = error;
				return error;
			
			case RELU_ACTIVATION:
				this.error = error;//.d_relu();
				return parameters[0].transpose().dot(this.error);
			
			case SOFTMAX_ACTIVATION:
				this.error = error;
				return parameters[0].transpose().dot(this.error);
			
			default:
				return null;
		}
	}
	
	public void calculateChanges() {
		deltaPara[1].sum(error);
		deltaPara[0].sum(error.dot(activation.transpose()));
	}
	
}
