package nnet.network.net;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import nnet.algebra.Matrix;
import nnet.network.Forwardable;
import nnet.network.layer.Layer;

/**
* Net
* An abstract class for network structure in a bigger neural network
* @author Muti Kara
*/
public abstract class Net implements Forwardable<Matrix> {
	/** List of layers */
	ArrayList<Layer> layers = new ArrayList<>();
	
	/**
	 * Takes a matrix as an input and returns an matrix as output.
	 * Converts input matrix into an one length matrix array.
	 * Then applies forwardPropagation method of all layers with order.
	 * */
	@Override
	public Matrix forwardPropagation(Matrix inputs) {
		Matrix[] output = new Matrix[]{ inputs };
		for (int i = 0; i < layers.size(); i++) {
			output = layers.get(i).forwardPropagation(output);
		}
		return Matrix.flatten(output);
	}
	
	/**
	* Ranodmizes entire network
	* @param rate
	*/
	public void randomize(double rate) {
		for (int i = 0; i < layers.size(); i++) {
			layers.get(i).randomize(rate);
		}
	}
	
	public int size() {
		return layers.size();
	}
	
	public Layer getLayer(int index) {
		return layers.get(index);
	}
	
	public void setLayer(int index, Layer layer) {
		layers.set(index, layer);
	}
	
	/**
	* Code for adding a layer should be implmented 
	* @param type
	* @param layerDescriptor
	*/
	public abstract void addLayer(int type, int ... layerDescriptor);
	
	@Override
	public void read(Scanner in) {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void write(FileWriter out) {
		// TODO Auto-generated method stub
	}
	
	@Override
	public String toString() {
		String str = layers.size() + "\n";
		for(int i = 0; i < layers.size(); i++){
			str += "\n" + layers.get(i).toString();
		}
		return str;
	}
	
}
