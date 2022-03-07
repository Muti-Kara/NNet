package layer;

import algebra.*;
/**
* Fully Connected Layer
* @author Muti Kara
*/
public class FullyConnected {
	Matrix weight, bias;
	double learningRate = HyperParameters.LEARNING_RATE;
	
	public FullyConnected(int sizePrev, int sizeNext){
		weight = new Matrix(sizeNext, sizePrev);
		bias = new Matrix(sizeNext, 1);
		weight.randomize();
		bias.randomize();
	}
	
	public FullyConnected(FullyConnected other){
		weight = new Matrix(other.weight.getRow(), other.weight.getCol());
		bias = new Matrix(other.bias.getRow(), other.bias.getCol());
	}
	
	public Matrix goForward(Matrix input, boolean softmax){
		if(softmax)
			return MatrixTools.softmax(weight.dot(input).sum(bias));
		else
			return MatrixTools.func(weight.dot(input).sum(bias));
	}
	
	public void sub(FullyConnected other){
		weight.sub(other.getWeight().sProd(learningRate));
		bias.sub(other.getBias().sProd(learningRate));
	}

	public double getLearningrate() {
		return learningRate;
	}

	public Matrix getWeight() {
		return weight;
	}

	public Matrix getBias() {
		return bias;
	}

	public double getRegularization() {
		return MatrixTools.regularization(weight) + MatrixTools.regularization(bias);
	}
	
	public void setLearningrate(double learningrate) {
		this.learningRate = learningrate;
	}

	public void setWeight(Matrix weight) {
		this.weight = weight;
	}

	public void setBias(Matrix bias) {
		this.bias = bias;
	}

	@Override
	public String toString() {
		return "\n" + weight + "\n" + bias;
	}
}
