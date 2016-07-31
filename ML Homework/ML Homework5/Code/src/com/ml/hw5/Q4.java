/**
 * 
 */
package com.ml.hw5;

import com.ml.hw5.classifier.impl.NaiveBayesBernouli;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.stats.NaiveBayesModel;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q4 {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getData(ClassifierUtil.SPAMBASE_MISSING_TRAINING_DATA_FILE, 
				ClassifierUtil.SPAMBASE_MISSING_TRAINING_FEATURE_FILE);
		DataSet testData = DataInput.getData(ClassifierUtil.SPAMBASE_MISSING_TEST_DATA_FILE, 
				ClassifierUtil.SPAMBASE_MISSING_TRAINING_FEATURE_FILE);
		int featureSize = trainingData.getFeatures().size() -1;
		//DataInput.normalizeData(trainingData, testData);
		
		NaiveBayesBernouli classifier = new NaiveBayesBernouli(featureSize);
		NaiveBayesModel model = classifier.train(trainingData);
		
		
		double[] confusionMatrixTraining = new double[4];
		double trainingError = classifier.testModel(model, trainingData, confusionMatrixTraining, 1);
		double[] confusionMatrixTest = new double[4];
		double testError = classifier.testModel(model, testData, confusionMatrixTest, 1);
		System.out.println(trainingError+"  -  "+testError);
	}

}
