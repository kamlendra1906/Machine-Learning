import java.util.Collections;

import com.ml.hw2.classifier.RidgeRegressionJama;
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
public class Q1L2RegressionSpam {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		DataInput.normalizeData(dataSet, null);
		
		double lambda = 0.1;
		int totalFolds = 10;
		double accuracy = 0;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		
		Collections.shuffle(dataSet.getData());
		
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
		
			Matrix weightMatrix = RidgeRegressionJama.train(ClassifierUtil.prepareData(trainingData), lambda);
			double error = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weightMatrix, true);
			
			accuracy += error;
			System.out.println("Error in fold " + fold + ": " + error);
		}
		System.out.println("Average Error: " + accuracy / totalFolds + "     Lambda= " + lambda);
	}
}