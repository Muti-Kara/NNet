package neuralnet.network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import neuralnet.matrix.Matrix;

/**
* @author Muti Kara
*/
public class Trainer {
	private List<Integer> validation = new ArrayList<>();
	private List<Integer> data = new ArrayList<>();
	private Matrix[] inputs, answers;
	
	public Trainer(Matrix[] inputs, Matrix[] answers, double splitRatio) {
		this.inputs = inputs;
		this.answers = answers;
		
		List<Integer> pointers = new ArrayList<>();
		int i = 0;
		for(; i < inputs.length; i++)
			pointers.add(i);
		
		Collections.shuffle(pointers);
		
		i = 0;
		for(; i < inputs.length * splitRatio; i++)
			data.add(pointers.get(i));
		
		for(; i < inputs.length; i++)
			validation.add(pointers.get(i));
	}
	
	public void shuffleData() {
		Collections.shuffle(data);
	}
	
	public Matrix dataInput(int index) {
		return inputs[data.get(index)];
	}
	
	public Matrix validInput(int index) {
		return inputs[validation.get(index)];
	}
	
	public Matrix dataAnswer(int index) {
		return answers[data.get(index)];
	}
	
	public Matrix validAnswer(int index) {
		return answers[validation.get(index)];
	}
}
