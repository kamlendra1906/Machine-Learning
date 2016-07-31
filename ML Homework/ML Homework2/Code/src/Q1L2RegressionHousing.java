import com.ml.hw2.classifier.RidgeRegressionJama;
import com.ml.hw2.data.DataInput;
import com.ml.hw2.data.DataSet;
import com.ml.hw2.util.ClassifierUtil;

import Jama.Matrix;

/**
 * 
 */

/**
 * @author Kamlendra Kumar
 *
 */
public class Q1L2RegressionHousing {

	public static void main(String[] args) throws Exception {

		DataSet trainingData = DataInput.getData(ClassifierUtil.HOUSING_TRAINING_FILE, ClassifierUtil.HOUSING_FEATURES_FILE);
		DataSet testData = DataInput.getData(ClassifierUtil.HOUSING_TEST_FILE, ClassifierUtil.HOUSING_FEATURES_FILE);
		
		DataInput.normalizeData(trainingData, testData);

		double bestTestError = Double.POSITIVE_INFINITY;
		double bestLambda = 0;
		double bestTrainingError = 0;
		
		double lambda = 0.000001;
		for (int i = 1; i < 15; i++) {
			lambda*= 10;
			Matrix weight = RidgeRegressionJama.train(ClassifierUtil.prepareData(trainingData), lambda);
			
			double testError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weight, true);
			double trainingError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(trainingData), weight, true);
			
			System.out.println("Lambda="+lambda+"     :Training Error="+trainingError+"    :Test Error="+testError);
			
			if(testError < bestTestError) {
				bestTestError = testError;
				bestLambda = lambda;
				bestTrainingError = trainingError;
			}
		}
		System.out.println("Best");
		System.out.println("Lambda="+bestLambda+"     :Training Error="+bestTrainingError+"    :Test Error="+bestTestError);
	}
}
