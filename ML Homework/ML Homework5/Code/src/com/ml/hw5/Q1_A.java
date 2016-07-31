/**
 * 
 */
package com.ml.hw5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ml.hw5.classifier.Classifier;
import com.ml.hw5.classifier.impl.AdaBoost;
import com.ml.hw5.classifier.impl.DecisionStump;
import com.ml.hw5.data.Data;
import com.ml.hw5.data.DataInput;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.model.AdaBoostModel;
import com.ml.hw5.model.DecisionStumpModel;
import com.ml.hw5.stats.AlphaClassifier;
import com.ml.hw5.stats.FeatureRank;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q1_A {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAMBASE_ORIGINAL_TRAINING_DATA_FILE, ClassifierUtil.SPAMBASE_ORIGINAL_FEATURE_FILE);
		int featureSize = dataSet.getFeatures().size() - 1;
		
		Map<String, Object> additionalData = new HashMap<String, Object>();
		additionalData.put(AdaBoost.GENERATE_ROUND_STATS, false);
		additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, false);
		additionalData.put(AdaBoost.GENERATE_CONFUSION_MATRIX, false);
		additionalData.put(AdaBoost.CLASSIFICATION_THRESHOLD, 0d);
		additionalData.put(AdaBoost.ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		
		Classifier classifier = new AdaBoost(featureSize, true, 100);
		classifier.trainModel(dataSet, dataSet, additionalData);
		
		List<FeatureRank> featureRanks = calculateFeatureRanks(classifier, dataSet);
		for(int i = 0 ; i < 20; i++) {
			System.out.print(featureRanks.get(i).getFeature()+" ,");
		}
	}

	@SuppressWarnings("unchecked")
	private static List<FeatureRank> calculateFeatureRanks(Classifier classifier, DataSet dataSet) throws Exception {
		int featureSize = dataSet.getFeatures().size() - 1;
		List<FeatureRank> list = new ArrayList<FeatureRank>();
		
		double[] gammaForFeature = new double[featureSize];
		List<AlphaClassifier>[] alphaClassifierListForFeature = new ArrayList[featureSize];
		generateStats(classifier, gammaForFeature, alphaClassifierListForFeature);
		
		double[] featureContribution = new double[featureSize];
		double totalDenominator = 0;
		
		for(Data data : dataSet.getData()) {
			for(int feature = 0; feature < featureSize; feature++) {
				featureContribution[feature] += getFeatureContributionForData(data, alphaClassifierListForFeature[feature]);
			}
			totalDenominator+= getDenominatorForData(classifier, data);
		}
		
		for(int i=0; i< featureSize; i++) {
			list.add(new FeatureRank(i, featureContribution[i]/totalDenominator));
		}
		Collections.sort(list);
		return list;
	}
	
	private static double getDenominatorForData(Classifier classifier, Data data) throws Exception {
		double value = 0;
		double labelValue = data.labelValue() == 0 ? -1 : 1;
		AdaBoost adaBoostClassifier = (AdaBoost) classifier;
		AdaBoostModel adaBoostModel = (AdaBoostModel) adaBoostClassifier.getAdaBoostModel();
		List<Double> decisionStumpWeights = adaBoostModel.getClassifierWeight();
		List<Classifier> decisionStumps = adaBoostModel.getClassifiers();
		
		for(int i=0; i< decisionStumpWeights.size(); i++) {
			value+= labelValue * decisionStumpWeights.get(i) * decisionStumps.get(i).classifyTestPoint(data, null);
		}
		
		return value;
	}

	private static double getFeatureContributionForData(Data data, List<AlphaClassifier> list) throws Exception {
		double featureContributionForData = 0;
		double labelValue = data.labelValue() == 0 ? -1 : 1;
		if(list != null && list.size() > 0) {
			for(AlphaClassifier alphaClassifier : list) {
				featureContributionForData+= labelValue * alphaClassifier.getAlpha() * alphaClassifier.getClassifier().classifyTestPoint(data, null);
			}
		}
		return featureContributionForData;
	}

	private static void generateStats(Classifier classifier, double[] gammaForFeature, List<AlphaClassifier>[] alphaClassifierListForFeature) {
		AdaBoost adaBoostClassifier = (AdaBoost) classifier;
		AdaBoostModel adaBoostModel = (AdaBoostModel) adaBoostClassifier.getAdaBoostModel();
		
		List<Double> decisionStumpWeights = adaBoostModel.getClassifierWeight();
		List<Classifier> decisionStumps = adaBoostModel.getClassifiers();
		
		int count=0;
		double totalAlpha = 0;
		for(Classifier decisionStump : decisionStumps) {
			DecisionStumpModel model = ((DecisionStump) decisionStump).getModel();
			int feature = model.getFeature();
			double alpha = decisionStumpWeights.get(count);
			gammaForFeature[feature] += alpha;
			totalAlpha += alpha;
			if(alphaClassifierListForFeature[feature] == null) {
				List<AlphaClassifier> list = new ArrayList<AlphaClassifier>();
				alphaClassifierListForFeature[feature] = list;
			} 
			alphaClassifierListForFeature[feature].add(new AlphaClassifier(alpha, decisionStump));
			count++;
		}
		ClassifierUtil.normalizeProbability(gammaForFeature, totalAlpha);
	}
}