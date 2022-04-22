package neuralnet.network.layer;

import java.io.FileWriter;
import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.Forwardable;

/**
* Layer
*/
public abstract class Layer implements Forwardable<Matrix[]>{

	@Override
	public Matrix[] forwardPropagation(Matrix[] inputs) {
		// TODO Auto-generated method stub
		return null;
	}
	
	public void randomize() {
		// TODO implement
	}

	@Override
	public void read(Scanner in) {
		// TODO Auto-generated method stub
	}

	@Override
	public void write(FileWriter out) {
		// TODO Auto-generated method stub
	}
	
}
