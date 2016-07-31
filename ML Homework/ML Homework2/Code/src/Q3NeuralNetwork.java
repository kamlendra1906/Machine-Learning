import java.util.ArrayList;
import java.util.List;

import com.ml.hw2.classifier.NeuralNetworkImpl;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q3NeuralNetwork {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		NeuralNetworkImpl nn = new NeuralNetworkImpl(9, 4, 8);
		List<double[]> trainingData = new ArrayList<double[]>();
		trainingData.add(new double[]{1,0,0,0,0,0,0,0});
		trainingData.add(new double[]{0,1,0,0,0,0,0,0});
		trainingData.add(new double[]{0,0,1,0,0,0,0,0});
		trainingData.add(new double[]{0,0,0,1,0,0,0,0});
		trainingData.add(new double[]{0,0,0,0,1,0,0,0});
		trainingData.add(new double[]{0,0,0,0,0,1,0,0});
		trainingData.add(new double[]{0,0,0,0,0,0,1,0});
		trainingData.add(new double[]{0,0,0,0,0,0,0,1});
		nn.train(trainingData, 0.000000001, 0.001);
		nn.test(trainingData);
		System.out.println("done");
	}

}
