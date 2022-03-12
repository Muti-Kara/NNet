package training;

import java.util.Random;

import algebra.NetworkParameters;
import algebra.matrix.Matrix;
import network.cnn.CNN;
import preproccess.images.InputImage;

/**
* Data organizer for ann
* @author Muti Kara
*/
public class DataOrganizer {	
	Matrix[] inputs = new Matrix[NetworkParameters.stochastic];
	int[] shuffled = new int[NetworkParameters.stochastic];
	
	InputImage images;
	Matrix answers;
	
	Random rand = new Random();
	
	public DataOrganizer(InputImage images) {
		answers = images.getAnswers();
		this.images = images;
		shuffle();
	}
	
	public void convert(CNN cnn) {
		for(int i = 0; i < NetworkParameters.stochastic; i++){
			inputs[i] = cnn.forwardPropagation( images.getInput(shuffled[i]) );
		}
	}
	
	public void shuffle() {
		for(int i = 0; i < shuffled.length; i++){
			shuffled[i] = rand.nextInt(NetworkParameters.dataSize);
		}
	}
	
	public Matrix getInput(int index) {
		return inputs[index];
	}
	
	public Matrix getAnswer(int index) {
		return answers.getVector( shuffled[index] );
	}
	
}