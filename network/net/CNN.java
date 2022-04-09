package neuralnet.network.net;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.network.layer.Convolutional;
import neuralnet.network.layer.Layer;
import neuralnet.network.layer.Pooling;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class CNN extends Network {
	int[] convolutional = NetworkOrganizer.convolutional;
	int[] kernel = NetworkOrganizer.kernel;
	int[] pool = NetworkOrganizer.pool;
	
	/**
	* Creates a convolutional network.
	 */
	public CNN() {
		layers = new Layer[2 * convolutional.length];
		for(int i = 0; i < 2 * convolutional.length; i += 2){
			layers[i] = new Convolutional(convolutional[i], kernel[i]);
			layers[i + 1] = new Pooling(pool[i]);
		}
	}
}
