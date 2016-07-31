/**
 * 
 */
package com.ml.hw7.classifier.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.ml.hw7.classifier.Classifier;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.stats.ActiveLearningDataStats;
import com.ml.hw7.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class ActiveLearning {
	
	private int featureSize;
	private int totalDataSize;
	private DataSet trainingDataSet;
	private DataSet testDataSet;
	
	public ActiveLearning(DataSet totalDataSet) throws Exception {
		this.totalDataSize = totalDataSet.dataSize();
		this.featureSize = totalDataSet.getFeatures().size() - 1;
		this.initializeTrainingAndTestData(totalDataSet);
	}

	private void initializeTrainingAndTestData(DataSet totalDataSet) throws Exception {
		trainingDataSet = new DataSet(totalDataSet.getLabelIndex(), totalDataSet.getFeatures());
		testDataSet = new DataSet(totalDataSet.getLabelIndex(), totalDataSet.getFeatures());
		List<Data> totalData = totalDataSet.getData();
		List<Data> trainingData = ClassifierUtil.randomSampleWithoutReplacement(totalData, 5);
		this.addDataToDataSet(trainingDataSet, trainingData);
		this.addDataToDataSet(testDataSet, totalData);
	}
	
	private void addDataToDataSet(DataSet dataset, List<Data> datas) throws Exception {
		for(Data data : datas) {
			dataset.addData(data);
		}
	}

	public List<double[]> runActiveLearning() throws Exception {
		Map<String, Object> additionalData = new HashMap<String, Object>();
		additionalData.put(AdaBoost.GENERATE_ROUND_STATS, false);
		additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, true);
		additionalData.put(AdaBoost.CLASSIFICATION_THRESHOLD, 0d);
		
		List<double[]> result = new ArrayList<double[]>();
		
		while(trainingDataSet.dataSize() <= totalDataSize/2) {
			Classifier classifier = new AdaBoost(this.featureSize, true, 1);
			classifier.trainModel(trainingDataSet, null, additionalData);
			
			List<ActiveLearningDataStats> stats = new ArrayList<ActiveLearningDataStats>();
			additionalData.put(AdaBoost.ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
			additionalData.put(AdaBoost.ACTIVE_LEARNING_STATS, stats);
			double error = classifier.testModel(testDataSet, additionalData);
			
			double[] roundResult = new double[2];
			roundResult[0] = (double)(trainingDataSet.dataSize() * 100) / totalDataSize;
			roundResult[1] = error;
			result.add(roundResult);
 			Collections.sort(stats);
			transferTestPointsToTrainingSet(stats);
			}
			return result;
		}

	private void transferTestPointsToTrainingSet(List<ActiveLearningDataStats> stats) throws Exception {
		Set<Data> testData = new HashSet<Data>(testDataSet.getData());
		int testDataSize = testData.size();
		int count = 0;
		int totalRun = testDataSize * 2/100;
		while(count <= totalRun) {
			Data point = stats.get(count).getData();
			trainingDataSet.addData(point);
			testData.remove(point);
			//System.out.println(count + "  " + totalRun);
			count++;
		}
		this.testDataSet = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		this.addDataToDataSet(testDataSet, new ArrayList<Data>(testData));
	}
}
