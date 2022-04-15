package neuralnet.network;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

import neuralnet.matrix.Matrix;
import neuralnet.network.layer.Layer;

/**
* @author Muti Kara
*/
public class NeuralNetwork implements Forwardable {
	ArrayList< Layer > layers = new ArrayList<>();
	double init;
	
	public NeuralNetwork(double init) {
		this.init = init;
	}
	
	public String classify(Matrix input) {
		Matrix ans = (Matrix) forwardPropagation(input);
		int max = 0;
		for(int r = 0; r < ans.getRow(); r++)
			if(ans.get(r, 0) > ans.get(max, 0))
				max = r;
		return (char) (max + 'A') + " " + ans.get(max, 0);
	}
	
	@Override
	public Object forwardPropagation(Object inputs) {
		Object output = ((Matrix) inputs).createClone();
		for(Layer layer : layers) {
			output = layer.forwardPropagation(output);
		}
	 	return output;
	}
	
	@Override
	public Object backPropagation(Object errors) {
		Object output = ((Matrix) errors).createClone();
		for(int i = layers.size() - 1; i >= 0; i--) {
			output = layers.get(i).backPropagation(output);
		}
	 	return (Matrix) output;
	}
	
	public void train(Trainer trainer, int epoch, int stochastic, double rate, double momentum) {
		while(epoch-->0) {
			trainer.shuffleData();
			for (int i = 0; i < stochastic; i++) {
				Matrix predictions = (Matrix) forwardPropagation(trainer.dataInput(i));
				backPropagation(predictions.sub(trainer.dataAnswer(i)));
				for(Layer layer : layers) {
					layer.calculateChanges();
				}
			}
			for (Layer layer : layers) {
				layer.applyChanges(rate, momentum);
			}
		}
	}
	
	@Override
	public String toString() {
		String str = "";
		for(Layer layer : layers) {
			str += layer.toString();
		}
		return str;
	}

	@Override
	public void read(Scanner in) {
		for(Layer layer : layers) {
			layer.read(in);;
		}
	}

	@Override
	public void write(FileWriter out) {
		for(Layer layer : layers) {
			layer.write(out);
		}
	}

	public NeuralNetwork addLayer(Layer newlayer) {
		layers.add(newlayer.randomize(init));
		return this;
	}

}
