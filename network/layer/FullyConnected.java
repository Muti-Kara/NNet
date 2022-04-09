package neuralnet.network.layer;

import neuralnet.algebra.matrix.*;

/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected extends Layer {
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
			case RELU_ACTIVATION:
				return new Matrix[]{ MatrixTools.relu(parameters[0].dot(input[0]).sum(bias)) };
			case SOFTMAX_ACTIVATION:
				return new Matrix[]{ MatrixTools.softmax(parameters[0].dot(input[0]).sum(bias)) };
			default:
				return null;
		}
	}
}
