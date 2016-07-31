/**
 * 
 */
package com.ml.hw4;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.ml.hw4.classifier.Classifier;
import com.ml.hw4.classifier.impl.AdaBoost;
import com.ml.hw4.data.Data;
import com.ml.hw4.data.DataInput;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q1ADABoost {

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
			Map<String, Object> additionalData = new HashMap<String, Object>();
			additionalData.put(AdaBoost.GENERATE_ROUND_STATS, true);
			additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, false);
			additionalData.put(AdaBoost.GENERATE_CONFUSION_MATRIX, true);
			
			Classifier classifier = new AdaBoost(featureSize, true, 100);
			classifier.trainModel(trainingData, testData, additionalData);
			
			if(fold == 0) {
				break;
			}
			
			/*double[] confusionMatrixTraining = new double[4];
			double trainingError = classifier.testModel(model, trainingData, confusionMatrixTraining, 1);
			double[] confusionMatrixTest = new double[4];
			double testError = classifier.testModel(model, testData, confusionMatrixTest, 1);
			
			totalTrainingError+= trainingError;
			totalTestError+= testError;
			System.out.println("Training Error in fold " + fold + ": " + trainingError);
			System.out.println("Test Error in fold " + fold + ": " + testError);
			System.out.println("Training Confusion Matrix:   " + ClassifierUtil.printArray(confusionMatrixTraining));
			System.out.println("Test Confusion Matrix:   " + ClassifierUtil.printArray(confusionMatrixTest)+ "\n");
			ClassifierUtil.updateAverageConfusionMatrix(averageTrainingConfusionMatrix, confusionMatrixTraining);
			ClassifierUtil.updateAverageConfusionMatrix(averageTestConfusionMatrix, confusionMatrixTest);*/
		}
		/*System.out.println("Average Training Error: " + totalTrainingError/totalFolds);
		System.out.println("Average Test Error: " + totalTestError/totalFolds);	
		ClassifierUtil.getAverage(averageTrainingConfusionMatrix, 10);
		ClassifierUtil.getAverage(averageTestConfusionMatrix, 10);
		System.out.println("Average Training Confusion Matrix: "+ ClassifierUtil.printArray(averageTrainingConfusionMatrix));
		System.out.println("Average Test Confusion Matrix: "+ ClassifierUtil.printArray(averageTestConfusionMatrix));*/

	}
}