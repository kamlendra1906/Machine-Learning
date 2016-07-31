/**
 * 
 */
package com.ml.hw4.classifier.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import com.ml.hw4.classifier.Classifier;
import com.ml.hw4.data.Data;
import com.ml.hw4.data.DataSet;
import com.ml.hw4.model.AdaBoostModel;
import com.ml.hw4.model.DecisionStumpModel;
import com.ml.hw4.model.Model;
import com.ml.hw4.stats.ActiveLearningDataStats;
import com.ml.hw4.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
@SuppressWarnings("unchecked")
public class AdaBoost implements Classifier {
	
	public static final String DATA_ERROR_WEIGHT = "DATA_ERROR_WEIGHT";
	public static final String CLASSIFICATION_THRESHOLD = "THRESHOLD";
	public static final String ALL_CONFUSION_MATRIX = "ALL_CONFUSION_MATRX";
	public static final String GENERATE_ROUND_STATS = "GENERATE_ROUND_STATS";
	public static final String GENERATE_ACTIVE_LEARNING_STATS = "GENERATE_ACTIVE_LEARNING_STATS";
	public static final String ACTIVE_LEARNING_STATS = "ACTIVE_LEARNING_STATS";
	public static final String GENERATE_CONFUSION_MATRIX = "GENERATE_CONFUSION_MATRIX";
	
	private int dataSize;
	private int featureSize;
	private int maxBoostingRound;
	private boolean optimalDedisionStump;
	
	private double[] dataErrorWeight;
	private Model adaBoostModel;
	
	public AdaBoost(int featureSize, boolean optimalDecisionStump, int maxBoostingRound) {
		this.maxBoostingRound = maxBoostingRound;
		this.adaBoostModel = new AdaBoostModel();
		this.optimalDedisionStump = optimalDecisionStump;
	}

	@Override
	public void trainModel(DataSet trainingDataSet, DataSet testDataSet,  Map<String, Object> additionalData) throws Exception {
		this.dataSize = trainingDataSet.dataSize();
		
		AdaBoostModel adaBoostModel = (AdaBoostModel)this.adaBoostModel;
		
		this.dataErrorWeight = initializeDataErrorWeight();
		additionalData.put(DATA_ERROR_WEIGHT, dataErrorWeight);
		
		int round = 0;
		while(adaBoostModel.getClassifiers().size() < this.maxBoostingRound) {
			
			Classifier classifier = new DecisionStump(this.optimalDedisionStump);
			if(this.optimalDedisionStump) {
				Map<Integer, Set<Double>> featureIdValueMap = getFeatureIdValuesMap(trainingDataSet);
				additionalData.put("hi", featureIdValueMap);
				
			}
			classifier.trainModel(trainingDataSet, testDataSet, additionalData);
			
			DecisionStumpModel decisionStumpModel = ((DecisionStump)classifier).getModel();
			double epsilon = decisionStumpModel.getRoundError();
			
			if(epsilon < 0.0005 || epsilon > 0.9995 || (Math.abs(0.5 - epsilon) < 0.01)) {
				continue;
			}
			//System.out.println("Decision Stump Trained "+round + "   "+classifier);
			adaBoostModel.getClassifiers().add(classifier);
			
			double alpha = calculateAlpha(epsilon);
			
			adaBoostModel.getClassifierWeight().add(alpha);
			recalculateDataErrorWeight(trainingDataSet, dataErrorWeight, alpha, classifier);
			
			boolean generateStats = (boolean) additionalData.get(GENERATE_ROUND_STATS);
			if(generateStats) {
				generateTrainingRoundStats(trainingDataSet, testDataSet, additionalData, round, decisionStumpModel, epsilon);
			}			
			round++;
		}
	}
	
	private Map<Integer, Set<Double>> getFeatureIdValuesMap(DataSet trainingDataSet) throws Exception {
		int featureSize = trainingDataSet.getFeatures().size() - 1;
		int dataSize = trainingDataSet.dataSize();
		
		Map<Integer, Set<Double>> featureIdValueMap = new HashMap<Integer, Set<Double>>();
		
		List<Data> dataSet = trainingDataSet.getData();
		for(int dataRow = 0; dataRow < dataSize; dataRow++) {
			Data data = dataSet.get(dataRow);
			
			for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
				double featureValue = data.getFeatureValue(featureIndex);
				Set<Double> featureUniqueValues = featureIdValueMap.get(featureIndex);
				if(featureUniqueValues == null) {
					featureUniqueValues = new TreeSet<Double>();
					featureUniqueValues.add(trainingDataSet.getFeatureMin()[featureIndex] - 1);
					featureUniqueValues.add(trainingDataSet.getFeatureMax()[featureIndex] + 1);
					featureIdValueMap.put(featureIndex, featureUniqueValues);
				}
				featureUniqueValues.add(featureValue);
			}
		}
		return featureIdValueMap;
	}
	
	private void generateTrainingRoundStats(DataSet trainingDataSet, DataSet testDataSet,  Map<String, Object> additionalData, int round, 
			DecisionStumpModel decisionStumpModel , double epsilon) throws Exception {
		double regularThreshold = 0;
		additionalData.put(ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		additionalData.put(CLASSIFICATION_THRESHOLD, regularThreshold);
		double trainingError = this.testModel(trainingDataSet, additionalData);
		
		additionalData.put(ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		additionalData.put(CLASSIFICATION_THRESHOLD, regularThreshold);
		double testError = this.testModel(testDataSet, additionalData);
		double auc = calculateAUC(testDataSet, additionalData);
		
		double[] stats = new double[7];
		stats[0] = round;
		stats[1] = decisionStumpModel.getFeature();
		stats[2] = decisionStumpModel.getThreshold();
		stats[3] = epsilon;
		stats[4] = trainingError;
		stats[5] = testError;
		stats[6] = auc;
		System.out.println(ClassifierUtil.printArray(stats));
		
	}

	private double calculateAUC(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		additionalData.put(ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		List<Double> thresholds = getThresholds(testDataSet, additionalData);
		
		double threshold = thresholds.get(0);
		double maxThreshold = thresholds.get(thresholds.size() - 1);
		double step = (maxThreshold - threshold) / 500;
		while(threshold <= maxThreshold) {
			additionalData.put(CLASSIFICATION_THRESHOLD, threshold);
			this.testModel(testDataSet, additionalData);
			threshold+= step;
		}
		List<double[]> confusionMatrixData = (List<double[]>) additionalData.get(ALL_CONFUSION_MATRIX);
		List<double[]> dataPoints = ClassifierUtil.getROCCurveData(confusionMatrixData);
		
		double totalSum = 0;
		
		for(int i=0; i < dataPoints.size() -1; i++) {
			int j = i+1;
			double[] dataI = dataPoints.get(i);
			double[] dataJ = dataPoints.get(j);
			totalSum += (dataJ[0] - dataI[0]) * (dataI[1] + dataJ[1]);
		}
		return Math.abs(totalSum/2);
	}

	private List<Double> getThresholds(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		Set<Double> thresholds = new TreeSet<Double>();
		for(Data data : testDataSet.getData()) {
			thresholds.add(getPredictedValueForData(data, additionalData));
		}
		return new ArrayList<Double>(thresholds);
	}

	/**
	 * Initialize the error weight for data elements
	 * @return
	 */
	private double[] initializeDataErrorWeight() {
		double[] dataErrorWeight = new double[this.dataSize];
		for(int i=0; i < dataSize; i++) {
			dataErrorWeight[i] = (double) 1/dataSize;
		}
		return dataErrorWeight;
	}

	@Override
	public double testModel(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		double totalError = 0;
		List<double[]> allConfusionMatrix = (List<double[]>) additionalData.get(ALL_CONFUSION_MATRIX);
		boolean generateConfusionMatrix = (boolean) additionalData.get(GENERATE_CONFUSION_MATRIX);
		double[] confusionMatrix = new double[4];
		for(Data data : testDataSet.getData()) {
			double actualLabel = data.labelValue() == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			double predictedLabel = classifyTestPoint(data, additionalData);
			if(generateConfusionMatrix) {
				ClassifierUtil.updateConfusionMatrix(confusionMatrix, actualLabel, predictedLabel);
			}
			if(actualLabel != predictedLabel) {
				totalError++;
			}
		}
		allConfusionMatrix.add(confusionMatrix);
		return totalError/testDataSet.dataSize();
	}

	@Override
	public double classifyTestPoint(Data dataPoint, Map<String, Object> additionalData) throws Exception {
		boolean generateActiveLearningStats = (boolean) additionalData.get(GENERATE_ACTIVE_LEARNING_STATS);
		
		double threshold = (double) additionalData.get(CLASSIFICATION_THRESHOLD);
		double predictedValue = getPredictedValueForData(dataPoint, additionalData);
		double predictedLabel =  predictedValue <= threshold ? Classifier.NON_SPAM : Classifier.SPAM;
		if(generateActiveLearningStats) {
			List<ActiveLearningDataStats> stats = (List<ActiveLearningDataStats>) additionalData.get(ACTIVE_LEARNING_STATS);
			ActiveLearningDataStats stat = new ActiveLearningDataStats();
			stat.setData(dataPoint);
			stat.setDistanceFromThreshold(Math.abs(threshold - predictedValue));
			stat.setPredictedLabel(predictedLabel);
			stats.add(stat);
		}
		return predictedLabel;
	}
	
	private double getPredictedValueForData(Data data, Map<String, Object> additionalData) throws Exception {
		AdaBoostModel adaBoostModel = (AdaBoostModel)this.adaBoostModel;
		List<Double> classifierWeight = adaBoostModel.getClassifierWeight();
		List<Classifier> classifiers = adaBoostModel.getClassifiers();
		
		double predictedValue = 0;
		int index = 0;
		
		for(Classifier classifier : classifiers) {
			predictedValue+= classifier.classifyTestPoint(data, additionalData) * classifierWeight.get(index);
			index++;
		}
		return predictedValue;
	}

	private double calculateAlpha(double epsilon) {
		double oneMinusEpsilon = 1 - epsilon;
		return 0.5d * Math.log(oneMinusEpsilon/epsilon);
	}

	private void recalculateDataErrorWeight(DataSet trainingDataSet, double[] dataErrorWeight, double alpha, Classifier classifier) throws Exception {
		int dataIndex = 0;
		double totalWeight = 0;
		for(Data data : trainingDataSet.getData()) {
			double actualLabel = data.labelValue() == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			double predictedLabel = classifier.classifyTestPoint(data, null);
			
			double oldDataErrorWeight = dataErrorWeight[dataIndex];
			double alphaYHt = alpha * actualLabel * predictedLabel;
			double newDataErrorWeight = oldDataErrorWeight * Math.exp(-alphaYHt);
			
			dataErrorWeight[dataIndex] = newDataErrorWeight;
			totalWeight+= newDataErrorWeight;
			dataIndex++;
		}
		ClassifierUtil.normalizeProbability(dataErrorWeight, totalWeight);
	}
	
	/**
	 * @return the dataSize
	 */
	public int getDataSize() {
		return dataSize;
	}

	/**
	 * @param dataSize the dataSize to set
	 */
	public void setDataSize(int dataSize) {
		this.dataSize = dataSize;
	}

	/**
	 * @return the featureSize
	 */
	public int getFeatureSize() {
		return featureSize;
	}

	/**
	 * @param featureSize the featureSize to set
	 */
	public void setFeatureSize(int featureSize) {
		this.featureSize = featureSize;
	}

	/**
	 * @return the dataErrorWeight
	 */
	public double[] getDataErrorWeight() {
		return dataErrorWeight;
	}

	/**
	 * @param dataErrorWeight the dataErrorWeight to set
	 */
	public void setDataErrorWeight(double[] dataErrorWeight) {
		this.dataErrorWeight = dataErrorWeight;
	}

	/**
	 * @return the adaBoostModel
	 */
	public Model getAdaBoostModel() {
		return adaBoostModel;
	}

	/**
	 * @param adaBoostModel the adaBoostModel to set
	 */
	public void setAdaBoostModel(Model adaBoostModel) {
		this.adaBoostModel = adaBoostModel;
	}
}