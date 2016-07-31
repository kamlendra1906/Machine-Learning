import java.util.Collections;

import com.ml.hw2.classifier.StochasticGradientDescentImpl;
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
public class Q1LinearRegressionGradiantDescentSpam {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);
		DataInput.normalizeData(dataSet, null);
		
		double lambda = 0.001;
		double threshold = 0.00001;
		int totalFolds = 10;
		double totalTestError = 0;
		double totalTrainingError = 0;
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
			
			StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
			Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , true);
			
			double trainingError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(trainingData), weight, true);
			double testError = ClassifierUtil.testWeight(ClassifierUtil.prepareData(testData), weight, true);
			
			totalTrainingError+= trainingError;
			totalTestError+= testError;
			System.out.println("Training Error in fold " + fold + ": " + trainingError);
			System.out.println("Test Error in fold " + fold + ": " + testError+"\n");
		}
		System.out.println("Average Training Error: " + totalTrainingError/totalFolds + "     Lambda= " + lambda);	
		System.out.println("Average Test Error: " + totalTestError/totalFolds + "     Lambda= " + lambda);	
	}
}