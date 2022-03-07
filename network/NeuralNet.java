package network;

import layer.FullyConnected;
import algebra.*;

/**
 * A FC network
 * @author Muti Kara
 * */
public class NeuralNet {
	int[] structure;
	FullyConnected[] layers;
	
	public NeuralNet() {
		this.structure = HyperParameters.structure;
		layers = new FullyConnected[structure.length];
		for(int i = 1; i < structure.length; i++){
			layers[i] = new FullyConnected(structure[i-1], structure[i]);
		}
	}
	
	public Matrix forwardPropagation(Matrix input) {
		Matrix output = input.createClone();
		for(int layer = 1; layer < structure.length; layer++) {
			output = layers[layer].goForward(output, layer == structure.length - 1);
		}
		return MatrixTools.softmax(output);
	}
	
	public FullyConnected getLayer(int index){
		return layers[index];
	}
	
	public int getSize(){
		return structure.length;
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
