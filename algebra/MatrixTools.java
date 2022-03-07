package algebra;

/**
* MatrixTools
* @author Muti Kara
*/
public class MatrixTools {
	
	/**
	* Applies ReLU activation function to all of the parameters.
	* @param matrix
	* @return matrix itself
	 */
	public static Matrix func(Matrix matrix){
		for(int r = 0; r < matrix.getRow(); r++) {
			for(int c = 0; c < matrix.getCol(); c++) {
				matrix.set(r, c, Math.max(0.5*(Math.exp(matrix.get(r, c)) - 1), matrix.get(r, c))); 
			}
		}
		return matrix;
	}
	
	/**
	* Applies ReLU activation function's derivative to all of the parameters.
	* @param matrix
	* @return matrix itself
	 */
	public static Matrix d_func(Matrix matrix) {
		for(int r = 0; r < matrix.getRow(); r++) {
			for(int c = 0; c < matrix.getCol(); c++) {
				matrix.set(r, c, (matrix.get(r, c) <= 0)? matrix.get(r, c) + 0.5 : 1);
			}
		}
		return matrix;
	}
	
	/**
	* Applies Softmax function to vector.
	* @param vector
	* @return vector itself
	 */
	public static Matrix softmax(Matrix vector){
		double sum = 0;
		for(int r = 0; r < vector.getRow(); r++){
			vector.set(r, 0, Math.exp(vector.get(r, 0)));
			sum += vector.get(r, 0);
		}
		for(int r = 0; r < vector.getRow(); r++){
			vector.set(r, 0, vector.get(r, 0) / sum);
		}
		return vector;
	}
	
	/**
	* Used for calculating regularization.
	* @param matrix
	* @return half of sum of squares of every parameters.
	 */
	public static double regularization(Matrix matrix) {
		double sum = 0;
		for(int j = 0; j < matrix.getRow(); j++){
			for(int i = 0; i < matrix.getCol(); i++){
				sum += matrix.get(j, i) * matrix.get(j, i);
			}
		}
		return sum / 2;
	}
	
	public static Matrix flatten(Matrix[] matArray){
		Matrix vector = new Matrix(matArray.length * matArray[0].getRow() * matArray[0].getCol(), 1);
		int k = 0;
		for(int i = 0; i < matArray.length; i++){
			for(int r = 0; r < matArray[i].getRow(); r++){
				for(int c = 0; c < matArray[i].getCol(); c++){
					vector.set(k++, 0, matArray[i].get(r, c));
				}
			}
		}
		return vector;
	}
	
	public static Matrix scale(Matrix matrix){
		double min = 1e4;
		for(int j = 0; j < matrix.getRow(); j++){
			min = Math.min(min, matrix.get(j, 0));
		}
		for(int j = 0; j < matrix.getRow(); j++){
			matrix.set(j, 0, (matrix.get(j, 0) - min));
		}
		return matrix;
	}
}



