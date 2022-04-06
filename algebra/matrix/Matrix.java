package neuralnet.algebra.matrix;

import java.util.Random;

/**
 * Matrix class for neural network
 * @author Muti Kara
 * */
public class Matrix {
	Random rand = new Random();
	double[][] matrix;
	int col, row;
	
	/**
	* Creates a zero matrix.
	* @param row
	* @param col
	 */
	public Matrix(int row, int col) {
		matrix = new double[row][col];
		this.col = col;
		this.row = row;		
	}
	
	/**
	* Creates a matrix object from given 2d array
	* @param matrix
	*/
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
		row = matrix.length;
		col = matrix[0].length;
	}
	
	/**
	* Sets the double value at given row and column.
	* @param row
	* @param col
	* @param value
	 */
	public void set(int row, int col, double value){
		matrix[row][col] = value;
	}
	
	/**
	* 
	* @param row
	* @param col
	* @return element at (row, col)
	 */
	public double get(int row, int col){
		return matrix[row][col];
	}
	
	/**
	* 
	* @return number of columns.
	 */
	public int getCol() {
		return col;
	}

	/**
	* 
	* @return number of rows.
	 */
	public int getRow() {
		return row;
	}
	
	/**
	* 
	* @return converts every element positive
	*/
	public Matrix abs() {
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix[r][c] = Math.abs(matrix[r][c]);
		return this;
	}
	
	/**
	* 
	* @return randomly fills the matrix.
	 */
	public Matrix randomize(double d) {
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix[r][c] = rand.nextGaussian() * d;
		return this;
	}
	
	/**
	* 
	* @return clones the matrix.
	 */
	public Matrix createClone() {
		Matrix C = new Matrix(row, col);
		for(int r = 0; r < row; r++) {
			C.matrix[r] = matrix[r].clone();
		}
		return C;
	}
	
	/**
	* 
	* @param row
	* @return the row of matrix as a vector.
	 */
	public Matrix getVector(int row) {
		Matrix C = new Matrix(col, 1);
		for(int r = 0; r < C.row; r++) {
			C.matrix[r][0] = matrix[row][r];
		}
		return C;
	}
	
	/**
	* 
	* @param B
	* @return dot product of matrices.
	 */
	public Matrix dot(Matrix B) {
		if(this.col != B.row) {
			System.out.println("Matrix dot error. Size Mismatch :" + col + " " + B.row);
			return null;
		}
		Matrix C = new Matrix(this.row, B.col);
		for(int crow = 0; crow < this.row; crow++) {
			for(int ccol = 0; ccol < B.col; ccol++) {
				double c = 0;
				for(int sum = 0; sum < this.col; sum++) {
					c += matrix[crow][sum] * B.matrix[sum][ccol];
				}
				C.matrix[crow][ccol] = c;
			}
		}
		return C;
	}
	
	/**
	* Summarizes two matrices element wise.
	* It happens in place. For convenience it returns this matrix.
	* @param B
	* @return this
	 */
	public Matrix sum(Matrix B) {
		if(!(row == B.row && col == B.col)) {
			System.out.println("Matrix sum error. Size Mismatch.");
			System.out.println("Row1: " + row + "\tRow2: " + B.row);
			System.out.println("Col1: " + col + "\tCol2: " + B.col);
			return null;
		}
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] += B.matrix[r][c];
			}
		}
		return this;
	}
	
	/**
	* Subtracts two matrices element wise.
	* It happens in place. For convenience it returns this matrix.
	* @param B
	* @return this
	 */
	public Matrix sub(Matrix B) {
		if(!(row == B.row && col == B.col)) {
			System.out.println("Matrix sub error. Size Mismatch.");
			return null;
		}
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] -= B.matrix[r][c];
			}
		}
		return this;
	}
	
	/**
	* Multiplies two matrices element wise.
	* It happens in place. For convenience it returns this matrix.
	* @param B
	* @return this
	 */
	public Matrix ewProd(Matrix B) {
		if(!(row == B.row && col == B.col)) {
			System.out.println("Matrix exProd error. Size Mismatch.");
		}
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] *= B.matrix[r][c];
			}
		}
		return this;
	}
	
	/**
	* Multiplies this matrix with a double b.
	* It happens in place. For convenience it returns this matrix.
	* @param B
	* @return this
	 */
	public Matrix sProd(double b) {
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] *= b;
			}
		}
		return this;
	}
	
	/**
	* Summarizes this matrix with a double b.
	* It happens in place. For convenience it returns this matrix.
	* @param B
	* @return this
	 */
	public Matrix sSum(double b) {
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] += b;
			}
		}
		return this;
	}
	
	/**
	* 
	* @return transpose of this matrix.
	 */
	public Matrix T() {
		Matrix C = new Matrix(col, row);
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				C.matrix[c][r] = matrix[r][c];
			}
		}
		return C;
	}
	
	@Override
	public String toString() {
		String str = "";
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++)
				str += matrix[r][c] + " ";
			str += "\n";
		}
		return str;
	}
	
}
