package neuralnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import neuralnet.algebra.Matrix;
import neuralnet.network.Forwardable;

/**
* Layer
*/
public abstract class Layer implements Forwardable<Matrix[]>{
	public int[] information;
	Matrix[] parameters;
	int type;
	
	public Layer(int type, int ... layerDescriptor) {
		this.type = type;
		this.information = layerDescriptor;
	}
	
	public void randomize(double rate) {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i].randomize(rate).abs();
		}
	}
	
	public Matrix getParameters(int index) {
		return parameters[index];
	}
	
	public void setParameters(int index, Matrix parameter) {
		this.parameters[index] = parameter;
	}

	public int size() {
		return this.parameters.length;
	}
	
	public int getType() {
		return type;
	}
	
	@Override
	public void read(Scanner in) {
		// TODO Auto-generated method stub
	}

	@Override
	public void write(FileWriter out) {
		// TODO Auto-generated method stub
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
