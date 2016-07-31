/**
 * 
 */
package com.ml.hw5;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.ml.hw5.classifier.Classifier;
import com.ml.hw5.classifier.impl.AdaBoost;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q1_B {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_DATA_FILE, ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_LABEL_FILE);
		DataSet testData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TEST_DATA_FILE, ClassifierUtil.SPAMBASE_POLLUTED_TEST_LABEL_FILE);
		
		int featureSize = trainingData.getFeatures().size() -1;
		
		Map<String, Object> additionalData = new HashMap<String, Object>();
		additionalData.put(AdaBoost.GENERATE_ROUND_STATS, false);
		additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, false);
		additionalData.put(AdaBoost.GENERATE_CONFUSION_MATRIX, false);
		additionalData.put(AdaBoost.CLASSIFICATION_THRESHOLD, 0d);
		additionalData.put(AdaBoost.ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		
		Classifier classifier = new AdaBoost(featureSize, false, 2500);
		classifier.trainModel(trainingData, testData, additionalData);
	}

}
