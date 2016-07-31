import com.ml.hw1.classifier.DecisionClassifier;
import com.ml.hw1.classifier.DecisionClassifierImpl;
import com.ml.hw1.classifier.TreeNode;
import com.ml.hw1.data.Data;
import com.ml.hw1.data.DataInput;
import com.ml.hw1.data.DataSet;

public class RegressionTreeQ1 {

	public static void main(String[] args) throws Exception {
		
		DataSet trainingData = DataInput.getDataFromFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\Housing\\housing_train.txt", 
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\Housing\\housing_features.txt", false, true);
		DataSet testData = DataInput.getDataFromFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\Housing\\housing_test.txt", 
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework1\\Data\\Housing\\housing_features.txt", false, false);
		
		DecisionClassifier classifier = new DecisionClassifierImpl(6, 15, 30);
		
		//DecisionClassifier classifier = new DecisionClassifierImpl(6, .15, 40);
		TreeNode root = classifier.buildClassifier(trainingData);
		testModel(testData, classifier, root, "Test Data");
		testModel(trainingData, classifier, root, "Training Data");
	}
	
	public static void testModel(DataSet testData, DecisionClassifier classifier, TreeNode root, String dataset) throws Exception {
		double diffSum = 0;
		for(Data data : testData.getData()){
			double perdictedValue = classifier.predict(root,data);
			diffSum += Math.pow((data.labelValue() - perdictedValue), 2);
		}
		double rmsd = diffSum/testData.getData().size();
		
		System.out.println("Mean squared error in "+dataset +":  "+ rmsd);

	}
}
