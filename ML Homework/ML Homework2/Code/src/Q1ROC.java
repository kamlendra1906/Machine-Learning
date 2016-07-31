import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
public class Q1ROC {

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

		List<double[]> confusionMatrixLinearRegressionData = new ArrayList<double[]>();
		List<double[]> confusionMatrixLogisticRegressionData = new ArrayList<double[]>();
		Matrix weightLinearRegression = null;
		Matrix weightLogisticRegression = null;
		DataSet trainingData = null;
		DataSet testData = null;
		for (int fold = 0; fold < totalFolds; fold++) {
			trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			for (int counter = 0; counter < dataSet.dataSize(); counter++) {
				if (counter >= fold * dataPerFold && counter < (fold + 1) * dataPerFold) {
					testData.addData(dataSet.getData().get(counter));
				} else {
					trainingData.addData(dataSet.getData().get(counter));
				}
			}

			//weightLinearRegression = ClassifierUtil.trainForLinearRegression(trainingData);
			weightLogisticRegression = ClassifierUtil.trainForLogisticRegression(trainingData);

			if (fold == 0) {
				break;
			}
		}
		double theta = 0;
		while (theta <= 1) {
			//confusionMatrixLinearRegressionData
				//	.add(ClassifierUtil.getConfusionMatrixForLinearRegression(weightLinearRegression, testData, theta));
			confusionMatrixLogisticRegressionData.add(
					ClassifierUtil.getConfusionMatrixForLogisticRegression(weightLogisticRegression, testData, theta));
			theta+= 0.001;
		}
		List<double[]> dataPoints = getROCCurveData(confusionMatrixLogisticRegressionData);
		for(double[] dataPoint : dataPoints) {
			System.out.println(ClassifierUtil.printArray(dataPoint));
		}
	}

	private static List<double[]> getROCCurveData(List<double[]> confusionMatrixData) {
		List<double[]> dataPoints = new ArrayList<double[]>();
		for(double[] confusionMatrix : confusionMatrixData) {
			dataPoints.add(getROCDataPoint(confusionMatrix));
		}
		return dataPoints;
	}

	private static double[] getROCDataPoint(double[] confusionMatrix) {
		double[] dataPoint = new double[2];
		double tp = confusionMatrix[ClassifierUtil.TRUE_POSITIVE];
		double fn = confusionMatrix[ClassifierUtil.FALSE_NEGATIVE];
		double fp = confusionMatrix[ClassifierUtil.FALSE_POSITIVE];
		double tn = confusionMatrix[ClassifierUtil.TRUE_NEGATIVE];
		
		double tpr = tp/(tp+fn);
		double fpr = fp/(fp+tn);
		dataPoint[0] = fpr;
		dataPoint[1] = tpr;
		return dataPoint;
	}
}
