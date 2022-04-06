package neuralnet.network.ann;

import neuralnet.network.ann.layer.FullyConnected;
import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN {
	int[] structure = NetworkOrganizer.structure;
	FullyConnected[] layers = new FullyConnected[structure.length];
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
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
			output = layers[layer].forwardPropagation(output, layer == structure.length - 1);
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
	
	/**
	* Sets index th layer with given FC layer
	* @param index
	 */
	public void setLayer(int index, FullyConnected layer){
		layers[index] = layer;
	}
	
	@Override
	public String toString() {
		String str = "";
		for(int i = 1; i < layers.length; i++){
			str += "\n" + layers[i].toString();
		}
		return str;
	}
}
