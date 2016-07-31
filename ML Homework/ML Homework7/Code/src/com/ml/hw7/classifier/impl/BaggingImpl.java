package com.ml.hw7.classifier.impl;

import java.util.ArrayList;
import java.util.List;

import com.ml.hw7.classifier.Classifier;
import com.ml.hw7.classifier.bagging.DecisionClassifier;
import com.ml.hw7.classifier.bagging.DecisionClassifierImpl;
import com.ml.hw7.classifier.bagging.TreeNode;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.util.ClassifierUtil;

public class BaggingImpl {

	private DataSet totalDataSet;
	private List<TreeNode> decisionTreeRoots;
	
	public BaggingImpl(DataSet totalDataSet) {
		this.totalDataSet = totalDataSet;
		decisionTreeRoots = new ArrayList<TreeNode>();
	}
	
	public void train(int baggingNum) throws Exception {
		for(int i=0; i<baggingNum; i++) {
			DataSet trainingDataSet = getTrainingDataSetForBagging();
			DecisionClassifier classifier  = new DecisionClassifierImpl(7, 0.001, 40);
			decisionTreeRoots.add(classifier.buildClassifier(trainingDataSet));
		}
	}
	
	public double test(DataSet testDataSet) throws Exception {
		DecisionClassifier classifier  = new DecisionClassifierImpl(7, 0.001, 40);
		double error = 0;
		for(Data data : testDataSet.getData()) {
			double predictedLabel = 0;
			double actualLabel = data.labelValue() == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			
			for(TreeNode tree : decisionTreeRoots) {
				double predictionByTree = classifier.predict(tree, data);
				predictedLabel+= predictionByTree == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			}
			predictedLabel = Math.signum(predictedLabel);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testDataSet.dataSize();
	}

	private DataSet getTrainingDataSetForBagging() throws Exception {
		DataSet trainingDataSet = new DataSet(totalDataSet.getLabelIndex(), totalDataSet.getFeatures());
		List<Data> trainingData = ClassifierUtil.randomSampleWithReplacement(totalDataSet.getData(), 100);
		addDataToDataSet(trainingDataSet, trainingData);
		return trainingDataSet;
	}
	
	private void addDataToDataSet(DataSet dataset, List<Data> datas) throws Exception {
		for(Data data : datas) {
			dataset.addData(data);
		}
	}
}
