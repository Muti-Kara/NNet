package neuralnet.network;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
// import neuralnet.network.net.ANN;
// import neuralnet.network.net.CNN;
import neuralnet.network.net.Net;

/**
* A feed forward network.
* @author Muti Kara
*/
public class NeuralNetwork implements Forwardable<Matrix> {
	ArrayList<Net> nets = new ArrayList<>();
	// CNN cnn;
	// ANN ann;
	
	/**
	* Takes two arguments: 1 CNN and 1 ANN
	* @param cnn
	* @param ann
	 */
	public NeuralNetwork(CNN cnn, ANN ann){
		this.cnn = cnn;
		this.ann = ann;
	}
	
	/**
	* 
	* @param input
	* @return Resulting matrix after forward propagating cnn and ann
	 */
	public Matrix forwardPropagation(Matrix input){
		return ann.forwardPropagation( cnn.forwardPropagation(input) );
	}
	
	/**
	* 
	* @param input
	* @return class of input
	 */
	public String classify(Matrix input){
		Matrix ans = forwardPropagation(input);
		int max = 0;
		for(int r = 0; r < ans.getRow(); r++)
			if(ans.get(r, 0) > ans.get(max, 0))
				max = r;
		return (char) (max + 'A') + " " + ans.get(max, 0);
	}
	
	@Override
	public String toString() {
		return cnn.toString() + "\n" + ann.toString();
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
