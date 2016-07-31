package com.ml.hw4;

import java.util.List;

import com.ml.hw4.classifier.impl.ActiveLearning;
import com.ml.hw4.data.DataInput;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.util.ClassifierUtil;

public class Q3ActiveLearning {

	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		ActiveLearning activeLearning = new ActiveLearning(dataSet);
		List<double[]> result = activeLearning.runActiveLearning();
		for(double[] rs : result) {
			System.out.println(ClassifierUtil.printArray(rs));
		}
	}

}
