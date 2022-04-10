package neuralnet.network;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import neuralnet.matrix.Matrix;
import neuralnet.network.net.Network;

/**
* A feed forward network.
* @author Muti Kara
*/
public class NeuralNetwork implements Forwardable<Matrix> {
	ArrayList<Network> nets = new ArrayList<>();
	
	/**
	* 
	* @param input
	* @return class of input
	 */
	public String classify(Matrix input) {
		Matrix ans = forwardPropagation(input);
		int max = 0;
		for(int r = 0; r < ans.getRow(); r++)
			if(ans.get(r, 0) > ans.get(max, 0))
				max = r;
		return (char) (max + 'A') + " " + ans.get(max, 0);
	}
	
	/**
	* 
	* @param input
	* @return Resulting matrix after forward propagating cnn and ann
	 */
	@Override
	public Matrix forwardPropagation(Matrix inputs) {
		Matrix output = inputs.createClone();
		for(Network net : nets) {
			output = net.forwardPropagation(output);
		}
	 	return output;
	}
	
	@Override
	public String toString() {
		String str = "";
		for(Network net : nets) {
			str += net.toString();
		}
		return str;
	}

	@Override
	public void read(Scanner in) {
		for(Network net : nets) {
			net.read(in);;
		}
	}

	@Override
	public void write(FileWriter out) {
		for(Network net : nets) {
			net.write(out);
		}
	}
	
	/**
	* Adds a new network
	* @param newNet
	* @return this neural network
	*/
	public NeuralNetwork addNet(Network newNet) {
		nets.add(newNet);
		return this;
	}
	
	/**
	* 
	* @param index
	* @return index th network
	*/
	public Network getNetwork(int index) {
		return nets.get(index);
	}
	
	/**
	* sets index th network
	* @param index
	* @param net
	*/
	public void setNetwork(int index, Network net) {
		nets.set(index, net);
	}
	
}
