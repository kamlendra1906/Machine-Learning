package com.ml.hw4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.ml.hw4.classifier.Classifier;
import com.ml.hw4.classifier.bagging.ECOCImpl;
import com.ml.hw4.classifier.impl.AdaBoost;
import com.ml.hw4.data.DataInput;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.util.ClassifierUtil;

public class Q4ECOC {

	public static void main(String[] args) throws Exception {

		Map<String, Object> additionalData = new HashMap<String, Object>();
		additionalData.put(AdaBoost.GENERATE_ROUND_STATS, false);
		additionalData.put(AdaBoost.ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, false);
		additionalData.put(AdaBoost.CLASSIFICATION_THRESHOLD, 0d);
		additionalData.put(AdaBoost.GENERATE_CONFUSION_MATRIX, false);
		
		DataSet testDataSet = DataInput.getDataForECOC(ClassifierUtil.ECOC_TEST_FILE);
		
		Classifier classifier = new ECOCImpl(8, 20);
		classifier.trainModel(null, null, additionalData);
		System.out.println("training done!");
		double error = classifier.testModel(testDataSet, additionalData);
		System.out.println(error);
	}

}
