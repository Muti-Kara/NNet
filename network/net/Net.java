package neuralnet.network.net;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.Forwardable;
import neuralnet.network.layer.Layer;

/**
* Net
*/
public abstract class Net implements Forwardable<Matrix> {
	ArrayList<Integer> structure = new ArrayList<>();
	ArrayList<Layer> layers = new ArrayList<>();
	
	@Override
	public Matrix forwardPropagation(Matrix inputs) {
		// TODO Auto-generated method stub
		return null;
	}
	
	public void randomize() {
		// TODO implement
	}
	
	public Layer getLayer(int index) {
		return layers.get(index);
	}
	
	public void setLayers(int index, Layer layer) {
		layers.set(index, layer);
	}
	
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
		// TODO Auto-generated method stub
		return null;
	}
	
}
