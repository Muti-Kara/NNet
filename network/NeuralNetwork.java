package nnet.network;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import nnet.algebra.Matrix;
import nnet.network.net.Net;

/**
* A feed forward network.
* @author Muti Kara
*/
public class NeuralNetwork implements Forwardable<Matrix> {
	ArrayList<Net> nets = new ArrayList<>();
	
	public void addNet(Net net) {
		nets.add(net);
	}
	
	/**
	* 
	* @param input
	* @return Resulting matrix after forward propagating cnn and ann
	 */
	public Matrix forwardPropagation(Matrix input){
		for (int i = 0; i < nets.size(); i++) {
			input = nets.get(i).forwardPropagation(input);
		}
		return input;
	}
	
	/**
	* 
	* @param input
	* @return class of input
	 */
	public String classify(double[][] input){
		Matrix ans = forwardPropagation(new Matrix(input));
		int max = 0;
		for(int r = 0; r < ans.getRowNum(); r++)
			if(ans.get(r, 0) > ans.get(max, 0))
				max = r;
		return (char) (max + 'A') + " ";// + ans.get(max, 0);
	}
	
	@Override
	public String toString() {
		String str = "";
		for (int i = 0; i < nets.size(); i++) {
			str += nets.get(i).toString();
		}
		return str;
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
