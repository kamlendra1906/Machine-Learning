package com.ml.hw4;

import com.ml.hw4.classifier.impl.GradiantBoostingImpl;
import com.ml.hw4.data.DataInput;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.util.ClassifierUtil;

public class Q7GradiantBoosting {

	public static void main(String[] args) throws Exception {
		DataSet trainingDataSet = DataInput.getData(ClassifierUtil.HOUSING_TRAINING_FILE, ClassifierUtil.HOUSING_FEATURE_FILE);
		DataSet testDataSet = DataInput.getData(ClassifierUtil.HOUSING_TEST_FILE, ClassifierUtil.HOUSING_FEATURE_FILE);
		
		GradiantBoostingImpl gradiantBoosting = new GradiantBoostingImpl(10);
		gradiantBoosting.trainModel(trainingDataSet);
		
		trainingDataSet = DataInput.getData(ClassifierUtil.HOUSING_TRAINING_FILE, ClassifierUtil.HOUSING_FEATURE_FILE);
		System.out.println("Mean squared error on Training :"+gradiantBoosting.testModel(trainingDataSet));
		System.out.println("Mean squared error on Test :"+gradiantBoosting.testModel(testDataSet));
	}
}