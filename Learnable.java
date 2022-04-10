package neuralnet;

import java.io.FileWriter;
import java.util.Scanner;

/**
* Readable
*/
public interface Learnable {
	public void read(Scanner in);
	public void write(FileWriter out);
}
