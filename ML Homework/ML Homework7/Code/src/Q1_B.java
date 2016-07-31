import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.CosineDistanceKernel;
import com.kami.hw7.svm.kernel.impl.GaussianKernel;
import com.kami.hw7.svm.kernel.impl.PolynomialKernel;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;
import com.ml.hw7.knn.KNNImpl;
import com.ml.hw7.knn.ranker.DistanceRanker;
import com.ml.hw7.knn.ranker.SimilarityRanker;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

public class Q1_B {

	private static SVMKernel kernel;
	private static Comparator<KNNNeighbour> ranker;
	private static Map<String, Object> parameters;
	
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getDigitData(ClassifierUtil.DIGIT_TRAINING_DATA_FILE_MODIFIED, 400);
		DataSet testData = DataInput.getDigitData(ClassifierUtil.DIGIT_TEST_DATA_FILE_MODFIED, 400);
		trainingData.setData(SVMUtil.sampleTrainingData(trainingData.getData(), 5));
		DataInput.normalizeData(trainingData, testData);
		
		setupForPolynomialKernel(trainingData.dataSize());
		KNNImpl knn = new KNNImpl(7, kernel, 2.5, true, ranker);
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
	
	private static void setupForPolynomialKernel(int dataSize) {
		kernel = new PolynomialKernel(dataSize);
		ranker = new SimilarityRanker();
		parameters = getAdditionalDataForPolynomialKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForPolynomialKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SVM_PARAMETER_ALPHA, .75d);
		parameters.put(SVMUtil.SVM_PARAMETER_BETA, 1d);
		parameters.put(SVMUtil.SVM_PARAMETER_DEGREE, 2d);
		return parameters;
	}

	private static void setupForGaussianKernel(int dataSize) {
		kernel = new GaussianKernel(dataSize);
		ranker = new SimilarityRanker();
		parameters = getAdditionalDataForGaussianKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForGaussianKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, .5);
		return parameters;
	}

	private static Map<String, Object> getAdditonalData() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		
		parameters.put(SVMUtil.SVM_PARAMETER_ALPHA, 0.001d);
		parameters.put(SVMUtil.SVM_PARAMETER_BETA, 0.01d);
		parameters.put(SVMUtil.SVM_PARAMETER_DEGREE, 2d);
		parameters.put(SVMUtil.SVM_PARAMETER_KERNEL_TYPE, SVMUtil.RBF_KERNEL);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, .5);
		return parameters;
	}
}