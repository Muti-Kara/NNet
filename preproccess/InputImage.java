package preproccess;

import java.io.IOException;

import algebra.HyperParameters;
import algebra.Matrix;

/**
* InputImage
*/
public class InputImage {
	int dataSize = HyperParameters.DATA_SIZE;
	int testSize = HyperParameters.TEST_SIZE;
	String projectDir = HyperParameters.projectDir;
	
	Matrix[] inputs = new Matrix[dataSize];
	Matrix[] tests = new Matrix[testSize];
	Matrix answers = new Matrix(dataSize, HyperParameters.structure[HyperParameters.structure.length - 1]);
	Matrix expected = new Matrix(testSize, HyperParameters.structure[HyperParameters.structure.length - 1]);
	
	public InputImage() throws IOException{
		for(int i = 0; i < dataSize; i++){
			ImageBuffer img = new ImageBuffer(projectDir + "/dataset/img" + (i+1) + ".jpeg");
			img.resize();
			img.turnBlackAndWhite();
			img.maximizeContrast();
			// img.write(HyperParameters.projectDir + "/outputs/imgTrain" + (i+1) + ".jpeg");
			inputs[i] = img.getMatrix();
			scaleInputs(inputs[i]);
			if(i < 2){
				answers.set(i, 0, 1);
			}else if(i < 4){
				answers.set(i, 1, 1);
			}else{
				answers.set(i, 2, 1);
			}
		}
		for(int i = 0; i < testSize; i++){
			ImageBuffer img = new ImageBuffer(projectDir + "/dataset/img" + (dataSize+i) + ".jpeg");
			img.resize();
			img.turnBlackAndWhite();
			img.maximizeContrast();
			// img.write(HyperParameters.projectDir + "/outputs/imgTest" + (i+1) + ".jpeg");
			tests[i] = img.getMatrix();
			scaleInputs(tests[i]);
			if(i < 1){
				expected.set(i, 0, 1);
			}else if(i < 2){
				expected.set(i, 1, 1);
			}else{
				expected.set(i, 2, 1);
			}
		}
	}
	
	public void scaleInputs(Matrix matrix){
		for(int r = 0; r < matrix.getRow(); r++)
			for(int c = 0; c < matrix.getCol(); c++)
				matrix.set(r, c, (matrix.get(r, c) == 0)? 0 : 1);
	}
	
	public Matrix getInputs(int index) {
		return inputs[index];
	}
	
	public Matrix getTests(int index) {
		return tests[index];
	}
	
	public Matrix getAnswers() {
		return answers;
	}
	
	public Matrix getExpected() {
		return expected;
	}
	
}
