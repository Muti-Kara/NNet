package neuralnet.network.ann.layer;

import neuralnet.algebra.matrix.*;

/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected {
	Matrix weight, bias;
	
	/**
	* Constructor takes two parameters:
	* @param previous layers size
	* @param next layers size
	 */
	public FullyConnected(int sizePrev, int sizeNext){
		weight = new Matrix(sizeNext, sizePrev);
		bias = new Matrix(sizeNext, 1);
	}
	
	/**
	* 
	* @return randomizes weights and biases
	*/
	public FullyConnected randomize(double rate) {
		weight.randomize(rate).abs();
		bias.randomize(rate).abs();
		return this;
	}
	
	/**
	* 
	* @param input
	* @param softmax
	* @return if softmax is true applies softmax activation otherwise applies ELU activation.
	 */
	public Matrix forwardPropagation(Matrix input, boolean softmax){
		if(softmax)
			return MatrixTools.softmax(weight.dot(input).sum(bias));
		return MatrixTools.relu(weight.dot(input).sum(bias));
	}
	
	/**
	* 
	* @return weight
	 */
	public Matrix getWeight() {
		return weight;
	}

	/**
	* 
	* @return bias
	 */
	public Matrix getBias() {
		return bias;
	}

	/**
	* 
	* @param weight
	 */
	public void setWeight(Matrix weight) {
		this.weight = weight;
	}

	/**
	* 
	* @param bias
	 */
	public void setBias(Matrix bias) {
		this.bias = bias;
	}

	@Override
	public String toString() {
		return "\n" + weight.getCol() + " " + weight.getRow() + "\n" + weight + "\n" + bias;
	}

}
