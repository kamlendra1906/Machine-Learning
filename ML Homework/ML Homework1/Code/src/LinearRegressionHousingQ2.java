import com.ml.hw1.classifier.LinearRegressionJama;
import com.ml.hw1.data.DataInput;
import com.ml.hw1.data.DataSet;
import com.ml.hw1.util.ClassifierUtil;

import Jama.Matrix;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class LinearRegressionHousingQ2 {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		DataSet trainingData = DataInput.getDataFromFile(
				"C:\\Users\\kkumar\\Desktop\\ML Homework1\\Data\\Housing\\housing_train.txt",
				"C:\\Users\\kkumar\\Desktop\\ML Homework1\\Data\\Housing\\housing_features.txt", true, true);
		DataSet testData = DataInput.getDataFromFile(
				"C:\\Users\\kkumar\\Desktop\\ML Homework1\\Data\\Housing\\housing_test.txt",
				"C:\\Users\\kkumar\\Desktop\\ML Homework1\\Data\\Housing\\housing_features.txt", true, false);

		Matrix weight  = LinearRegressionJama.train(ClassifierUtil.prepareData(trainingData));
		double error = LinearRegressionJama.test(ClassifierUtil.prepareData(testData), weight);
		System.out.println("Mean squared error in Test Data: "+ error);
		error = LinearRegressionJama.test(ClassifierUtil.prepareData(trainingData), weight);
		System.out.println("Mean squared error in Training Data: "+error);
	}

}
