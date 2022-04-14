package neuralnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import neuralnet.matrix.Matrix;
import neuralnet.network.Forwardable;

/**
* @author Muti Kara
*/
public abstract class Layer implements Forwardable<Matrix[]> {
	protected Matrix[] parameters;
	
	public Layer randomize(double d) {
		for(int i = 0; i < parameters.length; i++)
			parameters[i].randomize(d).abs();
		return this;
	}
	
	@Override
	public void read(Scanner in) {
		for(int i = 0; i < parameters.length; i++)
			parameters[i].read(in);
	}
	
	@Override
	public void write(FileWriter out) {
		for(int i = 0; i < parameters.length; i++)
			parameters[i].write(out);
	}
	
	@Override
	public String toString() {
		String str = "";
		for(int i = 0; i < parameters.length; i++) {
			str += parameters[i].toString() + "\n";
		}
		return str;
	}
	
	public abstract void applyChanges(double ... learningParameters);
	
	public Matrix getParameter(int index) {
		return parameters[index];
	}

	public void setParameter(int index, Matrix parameter) {
		parameters[index] = parameter;
	}

}
