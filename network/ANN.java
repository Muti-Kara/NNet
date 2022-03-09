package network;

import layer.FullyConnected;
import algebra.*;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN {
	int[] structure;
	FullyConnected[] layers;
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
		this.structure = HyperParameters.structure;
		layers = new FullyConnected[structure.length];
		for(int i = 1; i < structure.length; i++){
			layers[i] = new FullyConnected(structure[i-1], structure[i]);
		}
	}
	
	/**
	* 
	* @param input
	* @return output probabilities for categories
	 */
	public Matrix forwardPropagation(Matrix input) {
		Matrix output = input.createClone();
		for(int layer = 1; layer < structure.length; layer++) {
			output = layers[layer].goForward(output, layer == structure.length - 1);
		}
		return output;
	}
	
	/**
	* 
	* @param index
	* @return index th layer of network
	 */
	public FullyConnected getLayer(int index){
		return layers[index];
	}
	
	@Override
	public String toString() {
		String str = "" + structure.length + "\n";
		for(int i = 0; i < structure.length; i++){
			str += structure[i] + " ";
		}
		for(int i = 1; i < layers.length; i++){
			str += "\n" + layers[i].toString();
		}
		return str;
	}
}
