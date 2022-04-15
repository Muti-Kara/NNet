package neuralnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import neuralnet.matrix.Matrix;
import neuralnet.network.Forwardable;

/**
* @author Muti Kara
*/
public abstract class Layer implements Forwardable {
	public final static int TRAINMENT = -1;
	public final static int PRETRAINED = -2;
	
	protected Matrix[] parameters, deltaPara, prevDelta;
	boolean training;
	
	public Layer randomize(double d) {
		if (parameters == null)
			return this;
		
		for(int i = 0; i < parameters.length; i++)
			parameters[i].randomize(d).abs();
		
		return this;
	}
	
	public void applyChanges(double rate, double momentum) {
		for (int i = 0; i < parameters.length; i++) {
			parameters[i].sub(deltaPara[i].scalarProd(rate));
			parameters[i].sub(prevDelta[i].scalarProd(momentum));
			prevDelta[i] = deltaPara[i].createClone();
			deltaPara[i].scalarProd(0);
		}
	}
	
	public abstract void calculateChanges();
	
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
	
}
