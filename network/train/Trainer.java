package nnet.network.train;

import java.util.ArrayList;
import java.util.Collections;

import nnet.algebra.Matrix;
import nnet.network.net.Net;

/**
 * A trainer class for training networks.
 * This is an abstract class that provides funcitonality
 * of getting inputs shuffling them and spliting them 
 * into data, validation datasets.
 * @author Muti Kara
*/
public abstract class Trainer {
	protected ArrayList<Integer> valid = new ArrayList<>();
	protected ArrayList<Integer> data = new ArrayList<>();
	protected Matrix[] inputs, answers;
	
	protected Net net;
	
	protected int dataPtr = 0;
	protected int valPtr = 0;
	
	/**
	* Constructor takes a net for having a model, 
	* a matrix array for inputs and another array 
	* for answers, and a double value for splitting 
	* training and validation data.
	* @param net
	* @param inputs
	* @param answers
	* @param splitRatio
	*/
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
	
	/**
	* Training happens at three steps per epoch:
	* 	pre stochastic
	* 	during stochastic per data
	* 	and post stochastic
	* @param epoch
	* @param stochastic
	* @param rate
	* @param momentum
	*/
	public void train(int epoch, int stochastic, double rate, double momentum) {
		for (int i = 0; i < epoch; i++) {
			preStochastic(i, stochastic, momentum);
			for (int j = 0; j < stochastic; j++) {
				stochastic(j);
			}
			postStochastic(rate, momentum);
		}
	}
	
	/**
	* This abstract method will be called before iterating training data
	* @param atEpoch
	* @param stochastic
	* @param momentum
	*/
	public abstract void preStochastic(int atEpoch, int stochastic, double momentum);
	
	/**
	* This abstract method will be called during iteration of training data
	* @param at
	*/
	public abstract void stochastic(int at);
	
	/**
	* This abstract method will be called after iterating training data
	* @param rate
	* @param momentum
	*/
	public abstract void postStochastic(double rate, double momentum);
	
	public void shuffleData() {
		Collections.shuffle(data);
	}
	
	/**
	* 
	* @return matrix array of one input matrix and corresponding answer matrix
	*/
	public Matrix[] nextData() {
		if (dataPtr == data.size()) {
			dataPtr = 0;
			shuffleData();
		}
		return new Matrix[]{ inputs[data.get(dataPtr++)], answers[data.get(dataPtr++)] };
	}
	
	/**
	* 
	* @return matrix array of one input matrix and corresponding answer matrix
	*/
	public Matrix[] nextValid() {
		if (valPtr == valid.size()) {
			valPtr = 0;
		}
		return new Matrix[]{ inputs[valid.get(valPtr++)], answers[valid.get(valPtr++)] };
	}
	
	/**
	* 
	* @return best scored network
	*/
	public Net getBest() {
		return net;
	}
	
}
