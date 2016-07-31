import java.util.Collections;

import com.ml.hw1.classifier.DecisionClassifier;
import com.ml.hw1.classifier.DecisionClassifierImpl;
import com.ml.hw1.classifier.TreeNode;
import com.ml.hw1.data.Data;
import com.ml.hw1.data.DataInput;
import com.ml.hw1.data.DataSet;

public class DecisionTreeQ1 {

	public static void main(String[] args) throws Exception {
		DataSet dataSet = DataInput.getDataFromFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\SpamBase\\spambase.data",
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\SpamBase\\spambase.names", false, false);
		
		DecisionClassifier classifier = new DecisionClassifierImpl(7, 0.001, 40);
		
		int totalFolds = 10;
		double averageError = 0;
		double averagePercentageError = 0;
		int dataPerFold = dataSet.dataSize()/totalFolds;
		
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold<totalFolds; fold++) {
			DataSet trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			for(int counter = 0; counter < dataSet.dataSize(); counter++) {
				if(counter >= fold * dataPerFold && counter < (fold+1)*dataPerFold) {
					testData.addData(dataSet.getData().get(counter));
				} else {
					trainingData.addData(dataSet.getData().get(counter));
				}
			}
			TreeNode root = classifier.buildClassifier(trainingData);
			int error = 0;
			for(Data data : testData.getData()) {
				double predictedValue = classifier.predict(root, data);
				if(data.labelValue() != predictedValue) {
					error++;
				}
			}
			averagePercentageError += ((double)error)/dataPerFold*100;
			averageError += error;
			System.out.println(error+ " - "+ error);
			System.out.println(((double)error)/dataPerFold*100);
		}
		
		TreeNode node = classifier.buildClassifier(dataSet);
		System.out.println("Average Error: "+ averagePercentageError/totalFolds);
	}
}
