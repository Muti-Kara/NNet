package neuralnet.network.layer;

import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.Forwardable;

/**
* Layer
*/
public abstract class Layer implements Forwardable<Matrix[]> {
	Matrix[] parameters;
	Matrix bias;
	
	/**
	* 
	* @param d
	* @return randomizes parameters and bias
	*/
	public Layer randomize(double d) {
		bias.randomize(d).abs();
		for(int i = 0; i < parameters.length; i++)
			parameters[i].randomize(d).abs();
		return this;
	}
	
	@Override
	public void read(Scanner in) {
		bias.read(in);
		for(int i = 0; i < parameters.length; i++)
			parameters[i].read(in);
	}
	
	@Override
	public String toString() {
		String str = bias.toString() + "\n";
		for(int i = 0; i < parameters.length; i++) {
			str += parameters[i].toString() + "\n";
		}
		return str;
	}
	
	public Matrix getParameter(int index) {
		return parameters[index];
	}

	public void setParameter(int index, Matrix parameter) {
		parameters[index] = parameter;
	}

	public Matrix getBiases() {
		return bias;
	}

	public void setBiases(Matrix biases) {
		this.bias = biases;
	}
}
