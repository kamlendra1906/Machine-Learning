import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.perceptron.DualPerceptronImpl;
import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.GaussianKernel;
import com.kami.hw7.svm.kernel.impl.LinearKernel;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q3_B {

	private static SVMKernel kernel;
	private static Map<String, Object> parameters;

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet trainingData = DataInput.getData(ClassifierUtil.PERCEPTRON_TRAINING_FILE_NON_LINEAR, 
				ClassifierUtil.PERCEPTRON_FEATURE_FILE_NON_LINEAR);

		setupForLinearKernel(trainingData.dataSize());
		DualPerceptronImpl perceptron = new DualPerceptronImpl(kernel, trainingData.dataSize());
		perceptron.train(trainingData, parameters);
		double error = perceptron.testModel(trainingData, trainingData, parameters);
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
		parameters.put(SVMUtil.SVM_PARAMETER_NU, 1.75d);
		return parameters;
	}
	
	private static void setupForLinearKernel(int dataSize) {
		kernel = new LinearKernel(dataSize);
		parameters = getAdditionalDataForLinearKernel();
	}
	
	private static Map<String, Object> getAdditionalDataForLinearKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		return parameters;
	}
}
