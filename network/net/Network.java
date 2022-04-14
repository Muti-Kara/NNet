package neuralnet.network.net;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import neuralnet.matrix.Matrix;
import neuralnet.network.Forwardable;
import neuralnet.network.Trainer;
import neuralnet.network.layer.Layer;

/**
* @author Muti Kara
*/
public abstract class Network implements Forwardable<Matrix> {
	protected ArrayList< Layer > layers = new ArrayList<>();

	@Override
	public Matrix forwardPropagation(Matrix inputs) {
		Matrix[] preOutput = new Matrix[] { inputs.createClone() };
		for(Layer layer : layers) {
			preOutput = layer.forwardPropagation(preOutput);
		}
		return Matrix.flatten(preOutput);
	}
	
	@Override
	public void read(Scanner in) {
		for(Layer layer : layers){
			layer.read(in);
		}
	}
	
	@Override
	public void write(FileWriter out) {
		for(Layer layer : layers){
			layer.write(out);
		}
	}
	
	public int getDepth() {
		return layers.size();
	}
	
	abstract Network addLayer(int ... layerDescription);
	
	public Layer getLayer(int index) {
		return layers.get(index);
	}
	
	public void setLayers(int index, Layer layer) {
		layers.set(index, layer);
	}
	
	public void train(Trainer trainer, int epoch, int stochastic, double ... trainment) {
		while (epoch-->0) {
			trainer.shuffleData();
			for (int i = 0; i < stochastic; i++) {
				trainingEpochStep(trainer.dataInput(i), trainer.dataAnswer(i));
			}
			this.applyChanges(trainment);
		}
	}
	
	public abstract void trainingEpochStep(Matrix input, Matrix answer);
	
	public void applyChanges(double ... trainment) {
		for (Layer layer : layers) {
			layer.applyChanges(trainment);
		}
	}
	
	@Override
	public String toString() {
		String str = layers.size() + "\n";
		for(Layer layer : layers){
			str += layer.toString() + "\n";
		}
		return str;
	}
}
