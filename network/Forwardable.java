package neuralnet.network;

import neuralnet.Learnable;

/**
* Forwardable
*/
public interface Forwardable<T> extends Learnable {
	public T forwardPropagation(T inputs);
}
