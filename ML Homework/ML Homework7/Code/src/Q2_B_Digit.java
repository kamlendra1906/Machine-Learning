import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.GaussianKernel;
import com.kami.hw7.svm.kernel.impl.PolynomialKernel;
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
public class Q2_B_Digit {
	
	private static SVMKernel kernel;
	private static Map<String, Object> parameters;

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getDigitData(ClassifierUtil.DIGIT_TRAINING_DATA_FILE_MODIFIED, 400);
		DataSet testData = DataInput.getDigitData(ClassifierUtil.DIGIT_TEST_DATA_FILE_MODFIED, 400);
		
		trainingData.setData(SVMUtil.sampleTrainingData(trainingData.getData(), 20));
		DataInput.normalizeData(trainingData, testData);
		
		setupForGaussianKernel(trainingData.dataSize());
		ParzenWindowsImpl knn = new ParzenWindowsImpl(kernel);
		double error = knn.runKNN(trainingData, testData, parameters);
		System.out.println(error);
	}
	
	private static void setupForGaussianKernel(int dataSize) {
		kernel = new GaussianKernel(dataSize);
		parameters = getAdditionalDataForGaussianKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForGaussianKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, .5d);
		return parameters;
	}
	
	private static void setupForPolynomialKernel(int dataSize) {
		kernel = new PolynomialKernel(dataSize);
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

}
