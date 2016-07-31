import java.util.Collections;

import src.com.ml.hw3.classifier.GDAClassifierImpl;
import src.com.ml.hw3.classifier.stats.GDAModel;
import src.com.ml.hw3.data.DataInput;
import src.com.ml.hw3.data.DataSet;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q1GDA {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		DataInput.normalizeData(dataSet, null);
		
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
			
			GDAModel gdaModel = GDAClassifierImpl.train(trainingData);
			
			boolean useCommonCovarriance = true;
			double[] confusionMatrixTraining = new double[4];
			double trainingError = GDAClassifierImpl.testModel(gdaModel, trainingData, useCommonCovarriance, confusionMatrixTraining,1 );
			double[] confusionMatrixTest = new double[4];
			double testError = GDAClassifierImpl.testModel(gdaModel, testData, useCommonCovarriance, confusionMatrixTest, 1);
			
			totalTrainingError+= trainingError;
			totalTestError+= testError;
			System.out.println("Training Error in fold " + fold + ": " + trainingError);
			System.out.println("Test Error in fold " + fold + ": " + testError);
			System.out.println("Training Confusion Matrix:   " + ClassifierUtil.printArray(confusionMatrixTraining));
			System.out.println("Test Confusion Matrix:   " + ClassifierUtil.printArray(confusionMatrixTest)+ "\n");
			ClassifierUtil.updateAverageConfusionMatrix(averageTrainingConfusionMatrix, confusionMatrixTraining);
			ClassifierUtil.updateAverageConfusionMatrix(averageTestConfusionMatrix, confusionMatrixTest);
		}
		System.out.println("Average Training Error: " + totalTrainingError/totalFolds);
		System.out.println("Average Test Error: " + totalTestError/totalFolds);	
		ClassifierUtil.getAverage(averageTrainingConfusionMatrix, 10);
		ClassifierUtil.getAverage(averageTestConfusionMatrix, 10);
		System.out.println("Average Training Confusion Matrix: "+ ClassifierUtil.printArray(averageTrainingConfusionMatrix));
		System.out.println("Average Test Confusion Matrix: "+ ClassifierUtil.printArray(averageTestConfusionMatrix));

	}
}
