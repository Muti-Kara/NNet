package neuralnet.network.train;

import java.util.ArrayList;
import java.util.Collections;

import neuralnet.algebra.Matrix;
import neuralnet.network.net.Net;

/**
* @author Muti Kara
*/
public abstract class Trainer {
	protected ArrayList<Integer> valid = new ArrayList<>();
	protected ArrayList<Integer> data = new ArrayList<>();
	protected Matrix[] inputs, answers;
	
	protected double error = 0;
	protected Net net;
	
	protected int dataPtr = 0;
	protected int valPtr = 0;
	
	
	public Trainer(Net net, Matrix[] inputs, Matrix[] answers, double splitRatio) {
		this.net = net;
		this.inputs = inputs;
		this.answers = answers;
		
		ArrayList<Integer> pointers = new ArrayList<>();
		int i = 0;
		for(; i < inputs.length; i++)
			pointers.add(i);
		
		Collections.shuffle(pointers);
		
		i = 0;
		for(; i < inputs.length * splitRatio; i++)
			data.add(pointers.get(i));
		
		for(; i < inputs.length; i++)
			valid.add(pointers.get(i));
	}
	
	public void train(int epoch, int stochastic, double rate, double momentum) {
		for (int i = 0; i < epoch; i++) {
			error = 0;
			for (int j = 0; j < stochastic; j++) {
				stochastic(i, j);
			}
			if(Double.isNaN(error))
				return;
			
			apply(rate, momentum);
		}
	}
	
	public abstract void stochastic(int epoch, int stochastic);
	public abstract void apply(double rate, double momentum);
	public abstract void calculateError(Matrix answer);
	
	public void shuffleData() {
		Collections.shuffle(data);
	}
	
	public Matrix[] nextData() {
		if (dataPtr == data.size()) {
			dataPtr = 0;
			shuffleData();
		}
		return new Matrix[]{ inputs[data.get(dataPtr++)], answers[data.get(dataPtr++)] };
	}
	
	public Matrix[] nextValid() {
		if (valPtr == valid.size()) {
			valPtr = 0;
		}
		return new Matrix[]{ inputs[valid.get(valPtr++)], answers[valid.get(valPtr++)] };
	}
	
	public Net getBest() {
		return net;
	}
	
}
