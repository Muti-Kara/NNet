package preproccess;

import algebra.Matrix;
import algebra.MatrixTools;

/**
* InputData
*/
public class InputData {
	Matrix[] inputs;
	Matrix answers;
	
	public InputData(Matrix[] inputs, Matrix answers) {
		this.inputs = inputs;
		this.answers = answers;
		for(int i = 0; i < inputs.length; i++)
			MatrixTools.scale(inputs[i]);
	}
	
	public Matrix[] getInputs() {
		return inputs;
	}
	
	public Matrix getAnswers() {
		return answers;
	}

}
