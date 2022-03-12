package network.cnn;

import network.cnn.layer.*;
import algebra.NetworkParameters;
import algebra.matrix.*;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class CNN {
	int[] convolutional = NetworkParameters.convolutional;
	int[] kernel = NetworkParameters.kernel;
	int[] pool = NetworkParameters.pool;
	Convolutional[] convLayers = new Convolutional[convolutional.length];
	Pooling[] poolLayers = new Pooling[pool.length];
	
	/**
	* Creates a convolutional network.
	 */
	public CNN() {
		for(int i = 0; i < convLayers.length; i++){
			convLayers[i] = new Convolutional(convolutional[i], kernel[i]);
			poolLayers[i] = new Pooling(pool[i]);
		}
	}
	
	/**
	* takes image as matrix input and returns a vector for ann.
	* @param input
	* @return vector for ann
	 */
	public Matrix forwardPropagation(Matrix input) {
		Matrix[] preOutput = new Matrix[1];
		preOutput[0] = input;
		
		for(int i = 0; i < convolutional.length; i++){
			preOutput = convLayers[i].goForward(preOutput);
			preOutput = poolLayers[i].goForward(preOutput);
		}
		
		return MatrixTools.flatten(preOutput);
	}
	
	public Convolutional getConvLayer(int index) {
		return convLayers[index];
	}
	
	public Pooling getPoolLayer(int index) {
		return poolLayers[index];
	}
	
	public void setConvLayer(int index, Convolutional layer) {
		convLayers[index] = layer;
	}
	
	public void setPoolLayer(int index, Pooling layer) {
		poolLayers[index] = layer;
	}
	
	@Override
	public String toString() {
		String str = "";
		str += convLayers.length + "\n";
		for(int i = 0; i < convLayers.length; i++){
			str += convLayers[i].toString() + "\n";
			str += poolLayers[i].toString();
		}
		return str;
	}
}
