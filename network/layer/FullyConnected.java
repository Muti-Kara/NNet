package neuralnet.network.layer;

import neuralnet.algebra.matrix.Matrix;

/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected extends Layer {
	public final static int RELU = 0;
	public final static int SOFTMAX = 1;
	
	int activation;
	
	/**
	* Constructor takes two parameters:
	* @param previous layers size
	* @param next layers size
	 */
	public FullyConnected(int sizePrev, int sizeNext, int activation){
		this.activation = activation;
		parameters = new Matrix[2];
		parameters[0] = new Matrix(sizeNext, sizePrev);
		parameters[1] = new Matrix(sizeNext, 1);
	}
	
	/**
	* 
	* @param input
	* @param softmax
	* @return if softmax is true applies softmax activation otherwise applies ELU activation.
	 */
	@Override
	public Matrix[] forwardPropagation(Matrix[] input){
		switch (activation) {
			case RELU:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).relu() };
			case SOFTMAX:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).softmax() };
			default:
				return null;
		}
	}

	@Override
	public String toString() {
		return "\n"
		+ parameters[0].getCol() + " " + parameters[0].getRow()
		+ "\n" + parameters[0]
		+ "\n" + parameters[1];
	}

}
