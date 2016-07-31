import java.util.List;

import Jama.Matrix;
import src.com.ml.hw3.classifier.EMAlgorithmImpl;
import src.com.ml.hw3.classifier.stats.GaussianModel;
import src.com.ml.hw3.classifier.stats.MixtureModel;
import src.com.ml.hw3.data.DataInput;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q3EM3Features {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		List<Matrix> trainingData = DataInput.getData(ClassifierUtil.EM_TRAINING_FILE_3_FEATURE);
		int featureDimension = trainingData.get(0).getRowDimension();
		
		GaussianModel model1 = new GaussianModel(featureDimension);
		GaussianModel model2 = new GaussianModel(featureDimension);
		GaussianModel model3 = new GaussianModel(featureDimension);
		GaussianModel[] models = new GaussianModel[]{model1, model2, model3};
		
		MixtureModel mixtureModel = new MixtureModel(models.length, trainingData.size());
		mixtureModel.setModels(models);
		
		EMAlgorithmImpl emAlgo = new EMAlgorithmImpl(mixtureModel, trainingData);
		emAlgo.runEMAlgorithm();
		System.out.println(ClassifierUtil.printArray(model1.getCovariance().getRowPackedCopy()));
		System.out.println(ClassifierUtil.printArray(model1.getMean().getRowPackedCopy()));
		System.out.println("");
		System.out.println(ClassifierUtil.printArray(model2.getCovariance().getRowPackedCopy()));
		System.out.println(ClassifierUtil.printArray(model2.getMean().getRowPackedCopy()));
		System.out.println("");
		System.out.println(ClassifierUtil.printArray(model3.getCovariance().getRowPackedCopy()));
		System.out.println(ClassifierUtil.printArray(model3.getMean().getRowPackedCopy()));
	}

}
