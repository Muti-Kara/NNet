package network.ann.layer;

import algebra.matrix.*;

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
		weight.randomize();
		bias.randomize();
	}
	
	/**
	* Constructor takes another FullyConnected layer.
	* Copies its size, and creates a new FullyConnected layer.
	* @param other
	 */
	public FullyConnected(FullyConnected other){
		weight = new Matrix(other.weight.getRow(), other.weight.getCol());
		bias = new Matrix(other.bias.getRow(), other.bias.getCol());
	}
	
	/**
	* 
	* @param input
	* @param softmax
	* @return if softmax is true applies softmax activation otherwise applies ELU activation.
	 */
	public Matrix goForward(Matrix input, boolean softmax){
		if(softmax)
			return MatrixTools.softmax(weight.dot(input).sum(bias));
		return MatrixTools.func(weight.dot(input).sum(bias));
	}
	
	/**
	* subtracts two fully connected layer from each other.
	* @param other
	 */
	public void sub(FullyConnected other, double factor){
		weight.sub(other.getWeight().sProd(factor));
		bias.sub(other.getBias().sProd(factor));
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
		return "\n" + weight.getCol() + " " + weight.getRow() + 
			"\n" + weight + "\n" + bias;
	}

}
