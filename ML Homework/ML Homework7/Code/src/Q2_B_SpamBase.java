import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.GaussianKernel;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.knn.ParzenWindowsImpl;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q2_B_SpamBase {
	
	private static SVMKernel kernel;
	private static Map<String, Object> parameters;

	/**
	 * @param args
	 * @throws Exception 
	 */
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
			//DataInput.normalizeData(trainingData, testData);
			
			setupForGaussianKernel(trainingData.dataSize());
			ParzenWindowsImpl knn = new ParzenWindowsImpl(kernel);
			double error = knn.runKNN(trainingData, testData, parameters);
			System.out.println("Error in fold:"+fold+"   "+ error);
			avgError+= error;
		}
		System.out.println("average error: "+avgError/totalFolds);
	}
	
	private static void setupForGaussianKernel(int dataSize) {
		kernel = new GaussianKernel(dataSize);
		parameters = getAdditionalDataForGaussianKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForGaussianKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, 1.9d);
		return parameters;
	}
}
