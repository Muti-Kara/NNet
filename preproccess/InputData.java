package preproccess;

import algebra.Matrix;

/**
* InputData
*/
public class InputData {
	Matrix[] inputs;
	Matrix answers;
	
	Matrix[] asked;
	Matrix expected;
	
	public InputData(Matrix[] inputs, Matrix[] asked, Matrix answers, Matrix expected) {
		this.inputs = inputs;
		this.asked = asked;
		this.answers = answers;
		this.expected = expected;
		scale(inputs);
		scale(asked);
	}
	
	public void scale(Matrix[] matrix){
		for(int i = 0; i < matrix.length; i++){
			double min = 1e4;
			for(int j = 0; j < matrix[i].getRow(); j++){
				min = Math.min(min, matrix[i].get(j, 0));
			}
			for(int j = 0; j < matrix[i].getRow(); j++){
				matrix[i].set(j, 0, (matrix[i].get(j, 0) - min));
			}
		}
	}
	
	public Matrix[] getInputs() {
		return inputs;
	}
	
	public Matrix getAnswers() {
		return answers;
	}

	public Matrix[] getAsked() {
		return asked;
	}

	public Matrix getExpected() {
		return expected;
	}
	
}
