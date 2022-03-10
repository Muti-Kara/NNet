package preproccess;

import algebra.matrix.*;

/**
* InputData
* @author Muti Kara
*/
public class InputData {
	Matrix[] inputs;
	Matrix answers;
	
	/**
	* Simple class for holding a matrix array and matrix.
	* Additionally it scales the matrices in matrix array.
	* @param inputs
	* @param answers
	 */
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
