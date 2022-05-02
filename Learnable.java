package nnet;

import java.io.FileWriter;
import java.util.Scanner;

/**
* Learnable
*/
public interface Learnable {
	public void read(Scanner in);
	public void write(FileWriter out);
}
