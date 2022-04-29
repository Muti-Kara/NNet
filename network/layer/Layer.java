package neuralnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.Forwardable;

/**
* Layer
*/
public abstract class Layer implements Forwardable<Matrix[]>{
	int[] information;
	Matrix[] parameters;
	
	public Layer(int ... layerDescriptor) {
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

	public abstract Layer createClone();
	
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
		String str = "";
		for(int i = 0; i < parameters.length; i++){
			str += parameters[i].toString() + "\n";
		}
		return str;
	}
	
}
