package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected extends Layer {
	public final static int IDENTITY_ACTIVATION = 0;
	public final static int RELU_ACTIVATION = 1;
	public final static int SOFTMAX_ACTIVATION = 2;
	int activation;
	
	/**
	* Constructor takes two parameters:
	* @param previous layers size
	* @param next layers size
	 */
	public FullyConnected(int sizePrev, int sizeNext, int activation){
		this.activation = activation;
		parameters = new Matrix[1];
		parameters[0] = new Matrix(sizeNext, sizePrev);
		bias = new Matrix(sizeNext, 1);
	}
	
	/**
	* 
	* @param input
	* @param softmax
	* @return if activation is RELU then applies ReLU, otherwise applies softmax activation
	 */
	public Matrix[] forwardPropagation(Matrix[] input){
		switch (activation) {
			case IDENTITY_ACTIVATION:
				return input;
			case RELU_ACTIVATION:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(bias).relu() };
			case SOFTMAX_ACTIVATION:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(bias).softmax() };
			default:
				System.out.println("Activation function is undefined!");
				return null;
		}
	}
	
	public Matrix getWeight() {
		return parameters[0];
	}
	
	public void setWeight(Matrix mat) {
		parameters[0] = mat;
	}
}
