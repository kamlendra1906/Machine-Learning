import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.ml.hw2.classifier.DecisionClassifier;
import com.ml.hw2.classifier.DecisionClassifierImpl;
import com.ml.hw2.classifier.StochasticGradientDescentImpl;
import com.ml.hw2.classifier.TreeNode;
import com.ml.hw2.data.Data;
import com.ml.hw2.data.DataInput;
import com.ml.hw2.data.DataSet;
import com.ml.hw2.util.ClassifierUtil;

import Jama.Matrix;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q1ConfusionMatrix {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		DataInput.normalizeData(dataSet, null);
		
		int totalFolds = 10;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		
		Collections.shuffle(dataSet.getData());
		
		List<double[]> confusionMatrixData = new ArrayList<double[]>();
		
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
			
			double theta = 0.5;
			confusionMatrixData.add(getConfusionMatrixForDecisionTree(fold, trainingData, testData));
			confusionMatrixData.add(ClassifierUtil.getConfusionMatrixForLinearRegression(fold, trainingData, testData, theta));
			confusionMatrixData.add(ClassifierUtil.getConfusionMatrixForLogisticRegression(fold, trainingData, testData, theta));
			
			if(fold == 0) {
				break;
			}
		}
		System.out.println("**********************Confusion Matrix******************************");
		for(double[] confusionMatrix : confusionMatrixData) {
			System.out.println(ClassifierUtil.printArray(confusionMatrix));
		}
		System.out.println("*********************************************************************");
	}

	private static double[] getConfusionMatrixForDecisionTree(int fold, DataSet trainingData, DataSet testData) throws Exception {
		DecisionClassifier classifier = new DecisionClassifierImpl(7, 0.001, 40);
		TreeNode root = classifier.buildClassifier(trainingData);
		
		double[] confusionMatrixData = new double[4];
				
		for(Data data : testData.getData()) {
			double actualValue = data.labelValue();
			double predictedValue = classifier.predict(root, data);
			ClassifierUtil.updateConfusionMatrix(confusionMatrixData, actualValue, predictedValue);
		}
		return confusionMatrixData;
	}
}