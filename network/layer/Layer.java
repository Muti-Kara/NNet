package nnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import nnet.algebra.Matrix;
import nnet.network.Forwardable;

/**
* Layer
* An abstract class for layer structures
* @author Muti Kara
*/
public abstract class Layer implements Forwardable<Matrix[]>{
	public int[] information;
	Matrix[] parameters;
	int type;
	
	/**
	* Takes one or more arguments. <br />
	* First argument refers to this layer's type <br />
	* Other arguments refers to this layer's properties
	* @param type
	* @param layerDescriptor
	*/
	public Layer(int type, int ... layerDescriptor) {
		this.type = type;
		this.information = layerDescriptor;
	}
	
	/**
	* Randomizes this layer
	* @param rate
	*/
	public void randomize(double rate) {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i].randomize(rate).abs();
		}
	}
	
	/**
	* 
	* @param index
	* @return parameter matrix with given index
	*/
	public Matrix getParameter(int index) {
		return parameters[index];
	}
	
	/**
	* sets parameter matrix with given index
	* @param index
	* @param parameter
	*/
	public void setParameter(int index, Matrix parameter) {
		this.parameters[index] = parameter;
	}

	/**
	* 
	* @return number of parameter matrices
	*/
	public int size() {
		return this.parameters.length;
	}
	
	/**
	* 
	* @return type of this layer
	*/
	public int getType() {
		return type;
	}
	
	@Override
	public void read(Scanner in) {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i].read(in);
		}
	}

	@Override
	public void write(FileWriter out) {
		for (Matrix parameter : parameters) {
			parameter.write(out);
		}
	}
	
	@Override
	public String toString() {
		String str = parameters.length + "\n";
		for(int i = 0; i < parameters.length; i++){
			str += parameters[i].toString() + "\n";
		}
		return str;
	}
	
}
