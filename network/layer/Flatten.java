package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* Flatten
*/
public class Flatten extends Layer {

	@Override
	public Matrix forwardPropagation(Object inputs) {
		if (inputs instanceof Matrix[])
			return Matrix.flatten((Matrix[]) inputs);
		return null;
	}

	@Override
	public Matrix[] backPropagation(Object errors) {
		Matrix error;
		if(errors instanceof Matrix)
			error = (Matrix) errors;
		else
			return null;
		
		return null;
	}

	@Override
	public void calculateChanges() {}
	
	@Override
	public String toString() {
		return "FLATTEN\n";
	}
	
}
