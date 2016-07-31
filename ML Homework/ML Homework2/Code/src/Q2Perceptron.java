import com.ml.hw2.classifier.PerceptronClassifierImpl;
import com.ml.hw2.data.DataInput;
import com.ml.hw2.data.DataSet;
import com.ml.hw2.util.ClassifierUtil;

import Jama.Matrix;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q2Perceptron {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getData(ClassifierUtil.PERCEPTRON_TRAINING_FILE, ClassifierUtil.PERCEPTRON_FEATURE_FILE);
		DataInput.normalizeData(trainingData, null);
		
		double lambda = 0.1;
		PerceptronClassifierImpl perceptronImpl = new PerceptronClassifierImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = perceptronImpl.train(ClassifierUtil.prepareData(trainingData));
		System.out.println(ClassifierUtil.printArray(weight.getRowPackedCopy()));
		double w0 = weight.get(0, 0)*-1;
		for(int row=0; row < weight.getRowDimension(); row++) {
			for(int col=0; col < weight.getColumnDimension(); col++) {
				weight.set(row, col, weight.get(row, col)/w0);
			}
		}
		System.out.println(ClassifierUtil.printArray(weight.getRowPackedCopy()));
	}
}