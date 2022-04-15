package neuralnet.network.layer;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class Pooling extends Layer {
	int size;
	
	public Pooling(int size){
		this.size = size;
		parameters = new Matrix[0];
	}
	
	public Matrix[] forwardPropagation(Object inputs) {
		Matrix[] input;
		if (inputs instanceof Matrix[])
			input = ((Matrix[]) inputs);
		else
			return null;
		
		Matrix[] ans = new Matrix[input.length];
		for(int i = 0; i < input.length; i++){
			ans[i] = new Matrix(input[i].getRow() / size + 1, input[i].getCol() / size + 1);
			for(int r = 0; r < ans[i].getRow(); r++){
				for(int c = 0; c < ans[i].getCol(); c++){
					ans[i].set(r, c, -5e8);
				}
			}
			
			for(int r = 0; r < input[i].getRow(); r++){
				for(int c = 0; c < input[i].getCol(); c++){
					ans[i].set(r/size, c/size, Math.max(input[i].get(r, c), ans[i].get(r/size, c/size)));
				}
			}
		}
		
		return ans;
	}

	@Override
	public Matrix[] backPropagation(Object errors) {
		Matrix[] error;
		if (errors instanceof Matrix[])
			error = (Matrix[]) errors;
		else
			return null;
		
		return null;
	}

	@Override
	public void calculateChanges() {}
	
	@Override
	public String toString() {
		return size + "\n";
	}
}
