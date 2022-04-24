package neuralnet.network.net;

import neuralnet.network.layer.Convolutional;
import neuralnet.network.layer.Pooling;
// import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class CNN extends Net {
	// int[] convolutional = NetworkOrganizer.convolutional;
	// int[] kernel = NetworkOrganizer.kernel;
	// int[] pool = NetworkOrganizer.pool;
	// Convolutional[] convLayers = new Convolutional[convolutional.length];
	// Pooling[] poolLayers = new Pooling[pool.length];
	
	/**
	* Creates a convolutional network.
	 */
	public CNN() {
		for(int i = 0; i < convLayers.length; i++){
			convLayers[i] = new Convolutional(convolutional[i], kernel[i]);
			poolLayers[i] = new Pooling(pool[i]);
		}
	}
	
	// /**
	// * takes image as matrix input and returns a vector for ann.
	// * @param input
	// * @return vector for ann
	//  */
	// public Matrix forwardPropagation(Matrix input) {
	// 	Matrix[] preOutput = new Matrix[1];
	// 	preOutput[0] = input;
		
	// 	for(int i = 0; i < convolutional.length; i++){
	// 		preOutput = convLayers[i].forwardPropagation(preOutput);
	// 		preOutput = poolLayers[i].forwardPropagation(preOutput);
	// 	}
		
	// 	return Matrix.flatten(preOutput);
	// }
	
	// public Convolutional getConvLayer(int index) {
	// 	return convLayers[index];
	// }
	
	// public void setConvLayer(int index, Convolutional layer) {
	// 	convLayers[index] = layer;
	// }
	
	// @Override
	// public String toString() {
	// 	String str = "";
	// 	str += convLayers.length + "\n";
	// 	for(int i = 0; i < convLayers.length; i++){
	// 		str += convLayers[i].toString() + "\n";
	// 		str += poolLayers[i].toString();
	// 	}
	// 	return str;
	// }
}
