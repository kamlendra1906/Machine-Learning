/**
 * 
 */
package src.com.ml.hw3.classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.correlation.Covariance;

import Jama.Matrix;
import src.com.ml.hw3.classifier.stats.GDAModel;
import src.com.ml.hw3.data.Data;
import src.com.ml.hw3.data.DataSet;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class GDAClassifierImpl {

	public static GDAModel train(DataSet trainingDataSet) throws Exception {
		if (trainingDataSet == null) {
			throw new Exception("training data is null");
		}

		int featureSize = trainingDataSet.getFeatures().size() - 1;
		int trainingDataSize = trainingDataSet.getData().size();

		DataSet datasetClassZero = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		DataSet datasetClassOne = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		
		Map<Double, DataSet> classDataSetMap = new HashMap<Double, DataSet>();
		classDataSetMap.put((double) 0, datasetClassZero);
		classDataSetMap.put((double) 1, datasetClassOne);
		
		for(Data data : trainingDataSet.getData()) {
			classDataSetMap.get(data.labelValue()).addData(data);
		}
		
		double[] featureMeanClassZero = datasetClassZero.getDataFeatureMean();
		double[] featureMeanClassOne = datasetClassOne.getDataFeatureMean();
		
		double[][] commonClassCovarianceData = trainingDataSet.getFeatureDataAsArray();
		double[][] zeroClassCovarianceData = datasetClassZero.getFeatureDataAsArray();
		double[][] oneClassCovarianceData = datasetClassOne.getFeatureDataAsArray();
		
		GDAModel model = new GDAModel();
		
		model.setCoVarriance(new Matrix(new Covariance(commonClassCovarianceData).getCovarianceMatrix().getData(), featureSize, featureSize));
		model.setCoVarrianceClassZero(new Matrix(new Covariance(zeroClassCovarianceData).getCovarianceMatrix().getData(), featureSize, featureSize));
		model.setCoVarrianceClassOne(new Matrix(new Covariance(oneClassCovarianceData).getCovarianceMatrix().getData(), featureSize, featureSize));
		
		model.setMueZero(new Matrix(featureMeanClassZero, featureSize));
		model.setMueOne(new Matrix(featureMeanClassOne, featureSize));
		
		model.setProbabilityOfOne((double) datasetClassOne.dataSize() / trainingDataSize);
		
		return model;
	}
	
	public static double testModel(GDAModel model, DataSet testData, boolean commonCoVarriance, double[] confusionMatrix, double threshold) throws Exception {
		double totalError = 0;

		for (Data dataPoint : testData.getData()) {
			double actualClass = dataPoint.labelValue();
			double predictedClass = predictClass(model, dataPoint, commonCoVarriance, threshold);
			ClassifierUtil.updateConfusionMatrix(confusionMatrix, actualClass, predictedClass);
			if (actualClass != predictedClass) {
				totalError++;
			}
		}
		return totalError / testData.dataSize();
	}

	public static double predictClass(GDAModel model, Data dataPoint, boolean commonCoVarriance, double threshold) throws Exception {
		double probabilityOfClassZero = 1 - model.getProbabilityOfOne();
		int featureSize = dataPoint.getFeatures().size() - 1;
		Matrix x = preapreXMatrixFromData(dataPoint);
		
		Matrix coVarrianceZero = commonCoVarriance ? model.getCoVarriance() : model.getInvCoVarrianceClassZero();
		Matrix coVarrianceOne = commonCoVarriance ? model.getCoVarriance() : model.getInvCoVarrianceClassOne();
		
		
		double logLikelihoodOfZero = calculateLikelihood(probabilityOfClassZero, featureSize, coVarrianceZero, x, model.getMueZero(), commonCoVarriance);
		double logLikelihoodOfOne = calculateLikelihood(model.getProbabilityOfOne(), featureSize, coVarrianceOne, x, model.getMueOne(), commonCoVarriance);
		
		double likelihoodRatio = logLikelihoodOfOne/logLikelihoodOfZero;
		
		if(logLikelihoodOfOne > logLikelihoodOfZero) {
			return 1;
		}
		return 0;
	}
	
	public static List<Double> getThreshold(GDAModel model, DataSet testData, boolean commonCoVarriance) throws Exception {
		List<Double> thresholdList = new ArrayList<Double>();
		for(Data data : testData.getData()) {
			double probabilityOfClassZero = 1 - model.getProbabilityOfOne();
			int featureSize = data.getFeatures().size() - 1;
			Matrix x = preapreXMatrixFromData(data);
			
			Matrix coVarrianceZero = commonCoVarriance ? model.getCoVarriance() : model.getInvCoVarrianceClassZero();
			Matrix coVarrianceOne = commonCoVarriance ? model.getCoVarriance() : model.getInvCoVarrianceClassOne();
			
			
			double logLikelihoodOfZero = calculateLikelihood(probabilityOfClassZero, featureSize, coVarrianceZero, x, model.getMueZero(), commonCoVarriance);
			double logLikelihoodOfOne = calculateLikelihood(model.getProbabilityOfOne(), featureSize, coVarrianceOne, x, model.getMueOne(), commonCoVarriance);
			thresholdList.add(logLikelihoodOfOne/logLikelihoodOfZero);
		}
		return thresholdList;
	}

	private static Matrix preapreXMatrixFromData(Data dataPoint) throws Exception {
		int featureSize = dataPoint.getFeatures().size() - 1;
		double[] featureValues = new double[featureSize];
		for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			featureValues[featureIndex] = dataPoint.getFeatureValue(featureIndex);
		}
		return new Matrix(featureValues, featureSize);
	}

	private static double calculateLikelihood(double probabilityOfClass, int featureSize, Matrix coVarriance, Matrix x,
			Matrix classMean, boolean commonCoVarriance) {
		double logProbalityOfClass = ClassifierUtil.logValue(probabilityOfClass);
		double nBy2Log2Pie = ClassifierUtil.logValue(2 * Math.PI) * (featureSize / 2);
		double logCovarrianceDeterminnentBy2 = ClassifierUtil.logValue(coVarriance.det()) / 2;
		double exponentPart = calculateExponentPart(x, coVarriance, classMean, commonCoVarriance);
		return logProbalityOfClass - nBy2Log2Pie - logCovarrianceDeterminnentBy2 - exponentPart;
	}

	private static double calculateExponentPart(Matrix x, Matrix coVarriance, Matrix classMean, boolean commonCoVarriance) {
		Matrix xMinusMue = x.minus(classMean);
		if(commonCoVarriance) {
			return xMinusMue.transpose().times(coVarriance.inverse()).times(xMinusMue).det() / 2;
		}
		return xMinusMue.transpose().times(coVarriance).times(xMinusMue).det() / 2;
	}
}