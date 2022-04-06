package neuralnet.training;

import java.util.Random;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.cnn.CNN;
import neuralnet.preproccess.images.ImageOrganizer;

/**
* Data organizer for ann
* @author Muti Kara
*/
public class DataOrganizer {	
	Matrix[] inputs = new Matrix[NetworkOrganizer.stochastic];
	int[] shuffled = new int[NetworkOrganizer.stochastic];
	
	ImageOrganizer images;
	Matrix answers;
	
	Random rand = new Random();
	
	public DataOrganizer(ImageOrganizer images) {
		answers = images.getAnswers();
		this.images = images;
		shuffle();
	}
	
	public void convert(CNN cnn) {
		for(int i = 0; i < NetworkOrganizer.stochastic; i++){
			inputs[i] = cnn.forwardPropagation( images.getInput(shuffled[i]) );
		}
	}
	
	public void shuffle() {
		for(int i = 0; i < shuffled.length; i++){
			shuffled[i] = rand.nextInt(NetworkOrganizer.dataSize);
		}
	}
	
	public Matrix getInput(int index) {
		return inputs[index];
	}
	
	public Matrix getAnswer(int index) {
		return answers.getVector( shuffled[index] );
	}
	
}
