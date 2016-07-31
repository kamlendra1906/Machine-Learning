/**
 * 
 */
package src.com.ml.hw3.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import src.com.ml.hw3.classifier.stats.NaiveBayesHistogram4BinsModel;
import src.com.ml.hw3.data.Data;
import src.com.ml.hw3.data.DataSet;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class NaiveBayesHistogram9Bins {

	public static final double SPAM = 1.0d;
	public static final double NON_SPAM = 0.0d;

	private int featureSize;
	private int buckets;

	public NaiveBayesHistogram9Bins(int featureSize, int buckets) {
		this.featureSize = featureSize;
		this.buckets = buckets;
	}
	
	public NaiveBayesHistogram4BinsModel train(DataSet trainingDataSet) throws Exception {
		
		double[] featureMin = trainingDataSet.getFeatureMin();
		double[] featureMax = trainingDataSet.getFeatureMax();
		
		double[][] featureBucketValues = getFeatureBucket(featureMin, featureMax);
		
		DataSet spamDataSet = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		DataSet nonSPamDataSet = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		
		for(Data data : trainingDataSet.getData()) {
			double classLabel = data.labelValue();
			if(classLabel == SPAM) {
				spamDataSet.addData(data);
			} else {
				nonSPamDataSet.addData(data);
			}
		}
		
		double[][] featureBucketProbabilitiesSpam = new double[this.featureSize][this.buckets];
		double[][] featureBucketProbabilitiesNonSpam = new double[this.featureSize][this.buckets];
		
		Map<Double, double[][]> classFeatureProbabilityHistogramMap = new HashMap<Double, double[][]>();
		classFeatureProbabilityHistogramMap.put(SPAM, featureBucketProbabilitiesSpam);
		classFeatureProbabilityHistogramMap.put(NON_SPAM, featureBucketProbabilitiesNonSpam);
		
		this.populateClassConditionalProbabilities(classFeatureProbabilityHistogramMap, featureBucketValues, trainingDataSet);
		NaiveBayesHistogram4BinsModel model = new NaiveBayesHistogram4BinsModel(
				this.featureSize, spamDataSet.dataSize(), nonSPamDataSet.dataSize(), 9);
		model.setFeatureBucketValues(featureBucketValues);
		model.setFeatureBucketProbabilitiesSpam(featureBucketProbabilitiesSpam);
		model.setFeatureBucketProbabilitiesNonSpam(featureBucketProbabilitiesNonSpam);
		return model;
		
	}

	private void populateClassConditionalProbabilities(Map<Double, double[][]> classFeatureProbabilityHistogramMap,
			double[][] featureBuckets, DataSet trainingDataSet) throws Exception {
		for(Data data : trainingDataSet.getData()) {
			double classLabel = data.labelValue();
			double[][] classConditionalFeatureBinCount = classFeatureProbabilityHistogramMap.get(classLabel);
			for(int feature = 0; feature < featureSize; feature++) {
				double featureValue = data.getFeatureValue(feature);
				double[] bins = featureBuckets[feature];
				double[] binCount = classConditionalFeatureBinCount[feature];
				int bin = getBinForData(featureValue, bins);
				binCount[bin]++;
			}
		}
	}
	
	private int getBinForData(double featureValue, double[] bins) {
		int dataBin = -1;
		for(int bin=0; bin < this.buckets; bin++) {
			int nextBin = bin+1;
			if(bin == 0) {
				if(featureValue >= bins[bin] && featureValue <= bins[nextBin]) {
					dataBin = bin;
					break;
				}
			}
			if(bin == this.buckets -1) {
				if(featureValue > bins[bin] && featureValue <= bins[nextBin]) {
					dataBin = bin;
					break;
				}
			}
			if(featureValue > bins[bin] && featureValue <= bins[nextBin]) {
				dataBin = bin;
				break;
			}
		}
		return dataBin;
	}

	private double[][] getFeatureBucket(double[] featureMin, double[] featureMax) {
		double[][] featureBucketValues = new double[this.featureSize][this.buckets + 1];
		for(int feature = 0; feature < this.featureSize; feature++) {
			double[] featureBucket = new double[this.buckets + 1];
			populateFeatureBucketValues(featureBucket, featureMin[feature], featureMax[feature]);
			featureBucketValues[feature] = featureBucket;
		}
		return featureBucketValues;
	}

	private void populateFeatureBucketValues(double[] featureBucket, double featureMin, double featureMax) {
		double range = (featureMax - featureMin) / (this.buckets);
		for(int bin=0; bin < this.buckets + 1; bin++) {
			featureBucket[bin] = featureMin + range * bin;
		}
	}
	
	public double testModel(NaiveBayesHistogram4BinsModel model, DataSet testData, double[] confusionMatrix, double threshold) throws Exception {
		double totalError = 0;

		for (Data dataPoint : testData.getData()) {
			double actualClass = dataPoint.labelValue();
			double predictedClass = predictClass(model, dataPoint, threshold);
			ClassifierUtil.updateConfusionMatrix(confusionMatrix, actualClass, predictedClass);
			if (actualClass != predictedClass) {
				totalError++;
			}
		}
		return totalError / testData.dataSize();
	}
	
	public List<Double> getThreshold(NaiveBayesHistogram4BinsModel model, DataSet testData) {
		List<Double> thresholds = new ArrayList<Double>();
		for (Data dataPoint : testData.getData()) {
			double probabilityOfSpam = model.calculateProbabilityOfSpam(dataPoint);
			double probabilityOfNonSpam = model.calculateProbabilityOfNonSpam(dataPoint);
			thresholds.add(probabilityOfSpam/probabilityOfNonSpam);
		}
		Collections.sort(thresholds);
		return thresholds;
	}
	
	private double predictClass(NaiveBayesHistogram4BinsModel model, Data dataPoint, double threshold) {
		double probabilityOfSpam = model.calculateProbabilityOfSpam(dataPoint);
		double probabilityOfNonSpam = model.calculateProbabilityOfNonSpam(dataPoint);
		if(probabilityOfSpam / probabilityOfNonSpam > threshold) {
			return SPAM;
		}
		return NON_SPAM;
		/*if(probabilityOfNonSpam >= probabilityOfSpam) {
			return NON_SPAM;
		}
		return SPAM;*/
	}	
}
