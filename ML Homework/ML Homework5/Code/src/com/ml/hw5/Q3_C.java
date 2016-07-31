/**
 * 
 */
package com.ml.hw5;

import com.ml.hw5.classifier.bagging.L2BatchGradientDescentImpl;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.util.ClassifierUtil;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class Q3_C {

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
		
		double learningRate = 10;
		
		L2BatchGradientDescentImpl gradientDescent = new L2BatchGradientDescentImpl(learningRate, trainingData.getFeatures().size(), .5);
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), false);
		
		double trainingError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(trainingData), weight, false);
		double testError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weight, false);
		System.out.println(trainingError + "  "+testError);

	}

}
