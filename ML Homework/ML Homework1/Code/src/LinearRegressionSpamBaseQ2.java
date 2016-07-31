import java.util.Collections;

import com.ml.hw1.classifier.LinearRegressionJama;
import com.ml.hw1.data.DataInput;
import com.ml.hw1.data.DataSet;
import com.ml.hw1.util.ClassifierUtil;

import Jama.Matrix;

public class LinearRegressionSpamBaseQ2 {

	public static void main(String[] args) throws Exception {

		DataSet dataSet = DataInput.getDataFromFile(
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\SpamBase\\spambase.data",
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\SpamBase\\spambase.names", true, true);

		int totalFolds = 10;
		double meanSquaredError = 0;
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
			Matrix weightMatrix = LinearRegressionJama.train(ClassifierUtil.prepareData(trainingData));
			double error = LinearRegressionJama.test(ClassifierUtil.prepareData(testData), weightMatrix);
			meanSquaredError+= error;
			System.out.println("Error in fold "+fold+": "+error);
		}
		System.out.println("Average mean squared error: "+meanSquaredError/totalFolds);
	}
}