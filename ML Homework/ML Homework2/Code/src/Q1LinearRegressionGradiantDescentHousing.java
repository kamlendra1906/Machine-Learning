import java.util.Collections;

import com.ml.hw2.classifier.StochasticGradientDescentImpl;
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
public class Q1LinearRegressionGradiantDescentHousing {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getData(ClassifierUtil.HOUSING_TRAINING_FILE, ClassifierUtil.HOUSING_FEATURES_FILE);
		DataSet testData = DataInput.getData(ClassifierUtil.HOUSING_TEST_FILE, ClassifierUtil.HOUSING_FEATURES_FILE);
		DataInput.normalizeData(trainingData, testData);
		
		double lambda = 0.001;
		double threshold = 0.001;
		Collections.shuffle(trainingData.getData());
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , true);
		double testError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weight, true);
		double trainingError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(trainingData), weight, true);
		System.out.println("lambda=: "+lambda+"    testError="+testError+"     trainingError="+trainingError);
	}

}
