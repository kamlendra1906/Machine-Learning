import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.DistanceKernel;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;
import com.ml.hw7.knn.KNNImpl;
import com.ml.hw7.knn.ranker.DistanceRanker;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

public class Q1_A {
	
	private static SVMKernel kernel;
	private static Map<String, Object> parameters;
	private static Comparator<KNNNeighbour> ranker;

	public static void main(String[] args) throws Exception {

		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAMBASE_ORIGINAL_TRAINING_DATA_FILE, ClassifierUtil.SPAMBASE_ORIGINAL_FEATURE_FILE);
		
		int totalFolds = 10;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		double avgError = 0;
		
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
			DataInput.normalizeData(trainingData, testData);
			
			setupForDistanceKernel(trainingData.dataSize());
			KNNImpl knn = new KNNImpl(3, kernel, 2.5, true, ranker);
			double error = knn.runKNN(trainingData, testData, parameters);
			
			System.out.println("Error in fold:"+fold+"   "+ error);
			avgError+= error;
		}
		System.out.println("average error: "+avgError/totalFolds);
	}

	private static void setupForDistanceKernel(int dataSize) {
		kernel = new DistanceKernel(dataSize);
		parameters = getAdditonalData();
		ranker = new DistanceRanker();
	}
	
	private static Map<String, Object> getAdditonalData() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		return parameters;
	}
}