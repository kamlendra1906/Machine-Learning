/**
 * 
 */
package com.ml.hw4;

import java.util.Collections;

import com.ml.hw4.classifier.impl.BaggingImpl;
import com.ml.hw4.data.DataInput;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q6Bagging {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		
		int featureSize = dataSet.getFeatures().size() -1;
		int totalFolds = 10;
		double totalTestError = 0;
		double totalTrainingError = 0;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		
		Collections.shuffle(dataSet.getData());
		
		double[] averageTrainingConfusionMatrix = new double[4];
		double[] averageTestConfusionMatrix = new double[4];
		
		for (int fold = 0; fold < totalFolds; fold++) {
		
			DataSet trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			
			for (int counter = 0; counter < dataSet.dataSize(); counter++) {
				if (counter >= fold * dataPerFold && counter < (fold + 1) * dataPerFold) {
					testData.addData(dataSet.getData().get(counter));
				} else {
					trainingData.addData(dataSet.getData().get(counter));
				}
			}
			
			BaggingImpl bagging = new BaggingImpl(trainingData);
			bagging.train(50);
			double foldError = bagging.test(testData);
			totalTestError += foldError;
			System.out.println(foldError);
		}
		System.out.println(totalTestError / totalFolds);
	}
}
