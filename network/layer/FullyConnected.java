package nnet.network.layer;

import nnet.algebra.Matrix;

/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected extends Layer {
	public static final int INPUT = -1;
	public final static int RELU = 0;
	public final static int SOFTMAX = 1;
	
	/**
	* Constructor takes two parameters:
	* @param type 
	* @param description of this layer:<br />
	* 		number of neurons at previous layer<br />
	* 		number of neurons at this layer<br />
	* 		this layer's activation type (which is equals to type) <br />
	 */
	public FullyConnected(int type, int ... layerDescriptor){
		super(type, layerDescriptor);
		parameters = new Matrix[2];
		parameters[0] = new Matrix(information[1], information[0]);
		parameters[1] = new Matrix(information[1], 1);
	}
	
	/**
	* checks activation type of layer and then propagates forward
	* @param input
	* @return resulting matrix in an one length matrix array
	 */
	@Override
	public Matrix[] forwardPropagation(Matrix[] input){
		switch (information[2]) {
			case INPUT:
				return input;
			case RELU:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).relu() };
			case SOFTMAX:
				return new Matrix[]{ parameters[0].dot(input[0]).sum(parameters[1]).softmax() };
			default:
				return input;
		}
	}

}
