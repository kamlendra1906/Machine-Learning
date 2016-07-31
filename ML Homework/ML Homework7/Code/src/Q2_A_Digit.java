import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.CosineDistanceKernel;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;
import com.ml.hw7.knn.KNNImpl;
import com.ml.hw7.knn.ranker.DistanceRanker;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q2_A_Digit {

	private static SVMKernel kernel;
	private static Map<String, Object> parameters;
	private static Comparator<KNNNeighbour> ranker;

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getDigitData(ClassifierUtil.DIGIT_TRAINING_DATA_FILE_MODIFIED, 400);
		DataSet testData = DataInput.getDigitData(ClassifierUtil.DIGIT_TEST_DATA_FILE_MODFIED, 400);
		
		trainingData.setData(SVMUtil.sampleTrainingData(trainingData.getData(), 10));
		
		DataInput.normalizeData(trainingData, testData);

		setupForCosineDistance(trainingData.dataSize());
		KNNImpl knn = new KNNImpl(7, kernel, .02, false, ranker);
		double error = knn.runKNN(trainingData, testData, parameters);
		System.out.println(error);
	}
	
	private static void setupForCosineDistance(int dataSize) {
		kernel = new CosineDistanceKernel(dataSize);
		ranker = new DistanceRanker();
		parameters = getAdditionalDataForCosineDistanceKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForCosineDistanceKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		return parameters;
	}

}
