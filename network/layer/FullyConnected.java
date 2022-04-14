package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class FullyConnected extends Layer {
	public final static int TRAINMENT = -1;
	public final static int PRETRAINED = -2;
	public final static int IDENTITY_ACTIVATION = 0;
	public final static int RELU_ACTIVATION = 1;
	public final static int SOFTMAX_ACTIVATION = 2;
	
	Matrix[] deltaPara, prevDelta;
	Matrix activation, error;
	
	int activationType;
	boolean training;
	
	public FullyConnected(int previous, int ... layerDescription){
		this.activationType = layerDescription[1];
		this.training = layerDescription[2] == TRAINMENT;
		
		parameters = new Matrix[2];
		parameters[0] = new Matrix(layerDescription[0], previous);
		parameters[1] = new Matrix(layerDescription[0], 1);
		
		if (this.training) {
			deltaPara = new Matrix[2];
			deltaPara[0] = new Matrix(layerDescription[0], previous);
			deltaPara[1] = new Matrix(layerDescription[0], 1);
			
			prevDelta = new Matrix[2];
			prevDelta[0] = new Matrix(layerDescription[0], previous);
			prevDelta[1] = new Matrix(layerDescription[0], 1);
		}
	}
	
	@Override
	public Matrix[] forwardPropagation(Matrix[] input){
		if (training)
			activation = input[0];
		
		switch (activationType) {
			case IDENTITY_ACTIVATION:
				return input;
			
			case RELU_ACTIVATION:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).relu() };
			
			case SOFTMAX_ACTIVATION:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).softmax() };
			
			default:
				System.out.println("Activation function is undefined!");
				return null;
		}
	}
	
	public Matrix backPropagation(Matrix error) {
		switch (activationType) {
			case IDENTITY_ACTIVATION:
				return error;
			
			case RELU_ACTIVATION:
				this.error = error.d_relu();
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
	
	@Override
	public void applyChanges(double... learningParameters) {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i].sub(deltaPara[i].scalarProd(learningParameters[0]));
			parameters[i].sub(prevDelta[i].scalarProd(learningParameters[1]));
			prevDelta[i] = deltaPara[i];
			deltaPara[i].scalarProd(0);
		}
	}

}
