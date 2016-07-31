import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.GaussianKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataInput;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;
import com.ml.hw7.knn.KNNImpl;
import com.ml.hw7.knn.ReliefImpl;
import com.ml.hw7.knn.ranker.DistanceRanker;
import com.ml.hw7.knn.ranker.SimilarityRanker;
import com.ml.hw7.util.ClassifierUtil;
import com.ml.hw7.util.SVMUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q5 {
	
	private static SVMKernel kernel;
	private static Comparator<KNNNeighbour> ranker;
	private static Map<String, Object> parameters;

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAMBASE_ORIGINAL_TRAINING_DATA_FILE, ClassifierUtil.SPAMBASE_ORIGINAL_FEATURE_FILE);
		DataInput.normalizeData(dataSet, null);
		setupForGaussianKernel(dataSet.dataSize());
		ReliefImpl relief = new ReliefImpl(7, kernel, dataSet.getFeatures().size() -1, ranker);
		List<Integer> features = relief.runRelief(dataSet, parameters, 5);

		DataSet newDataSet = new DataSet(5, DataInput.getFeatures(5));
		for(Data data : dataSet.getData()) {
			Data newData = new Data(6);
			newData.setDataSet(newDataSet);
			int newFeature = 0; 
			for(Integer featureId : features) {
				newData.setFeatureValue(newFeature++, data.getFeatureValue(featureId));
				newData.setLabelValue(data.labelValue());
			}
			newDataSet.addData(newData);
		}
		
		KNNImpl knn = new KNNImpl(7, kernel, 2.5, true, ranker);
		double error = knn.runKNN(newDataSet, newDataSet, parameters);
		System.out.println(error);
	}

	
	private static void setupForGaussianKernel(int dataSize) {
		kernel = new GaussianKernel(dataSize);
		parameters = getAdditionalDataForGaussianKernel();
		ranker = new SimilarityRanker();
	}
	
	private static Map<String, Object> getAdditionalDataForGaussianKernel() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, 1.75d);
		return parameters;
	}
}
