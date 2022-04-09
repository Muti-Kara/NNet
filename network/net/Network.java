package neuralnet.network.net;

import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
import neuralnet.algebra.matrix.MatrixTools;
import neuralnet.network.Forwardable;
import neuralnet.network.layer.Layer;

/**
* Network
*/
public abstract class Network implements Forwardable<Matrix> {
	Layer[] layers;

	@Override
	public Matrix forwardPropagation(Matrix inputs) {
		Matrix[] preOutput = new Matrix[1];
		preOutput[0] = inputs.createClone();
		
		for(int i = 0; i < layers.length; i++) {
			preOutput = layers[i].forwardPropagation(preOutput);
		}
		
		return MatrixTools.flatten(preOutput);
	}
	
	public Layer getLayer(int index) {
		return layers[index];
	}
	
	public void setLayers(int index, Layer layer) {
		layers[index] = layer;
	}
	
	@Override
	public void read(Scanner in) {
		for(int i = 0; i < layers.length; i++){
			layers[i].read(in);
		}
	}
	
	@Override
	public String toString() {
		String str = layers.length + "\n";
		for(int i = 0; i < layers.length; i++){
			str += layers[i].toString() + "\n";
		}
		return str;
	}
}
