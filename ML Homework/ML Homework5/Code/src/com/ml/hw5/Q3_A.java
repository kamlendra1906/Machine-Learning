/**
 * 
 */
package com.ml.hw5;

import com.ml.hw5.classifier.bagging.StochasticGradientDescentImpl;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.util.ClassifierUtil;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class Q3_A {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_DATA_FILE,
				ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_LABEL_FILE);
		DataSet testData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TEST_DATA_FILE,
				ClassifierUtil.SPAMBASE_POLLUTED_TEST_LABEL_FILE);
		DataInput.normalizeData(trainingData, testData);
		
		double lambda = 0.1;
		double threshold = 0.00001;
		
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , false);
		
		double trainingError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(trainingData), weight, false);
		double testError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weight, false);
		System.out.println(trainingError + "  "+testError);
	}

}
