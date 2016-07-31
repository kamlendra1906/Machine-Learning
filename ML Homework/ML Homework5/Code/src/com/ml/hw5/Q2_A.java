/**
 * 
 */
package com.ml.hw5;

import com.ml.hw5.classifier.impl.NaiveBayesGaussianFeatureImpl;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.model.NaiveBayesGaussianFeatureModel;
import com.ml.hw5.util.ClassifierUtil;


/**
 * @author kkumar	
 *
 */
public class Q2_A {

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
		
		int featureSize = trainingData.getFeatures().size() - 1;
		
		NaiveBayesGaussianFeatureImpl classifier = new NaiveBayesGaussianFeatureImpl(featureSize);
		NaiveBayesGaussianFeatureModel model = classifier.train(trainingData);
		
		double[] confusionMatrixTraining = new double[4];
		double trainingError = classifier.testModel(model, trainingData, confusionMatrixTraining, 1);
		double[] confusionMatrixTest = new double[4];
		double testError = classifier.testModel(model, testData, confusionMatrixTest, 1);
		
		System.out.println(trainingError + "   "+testError);
	}

}
