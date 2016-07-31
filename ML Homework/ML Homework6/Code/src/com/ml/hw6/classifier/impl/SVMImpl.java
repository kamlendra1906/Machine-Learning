/**
 * 
 */
package com.ml.hw6.classifier.impl;

import java.util.Map;

import com.ml.hw6.classifier.Classifier;
import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.data.SVMData;
import com.ml.hw6.util.SVMUtil;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

/**
 * @author kkumar
 *
 */
public class SVMImpl implements Classifier {

	/* (non-Javadoc)
	 * @see com.ml.hw6.classifier.Classifier#trainModel(com.ml.hw6.data.DataSet, com.ml.hw6.data.DataSet, java.util.Map)
	 */
	
	private svm_model model;
	
	@Override
	public void trainModel(DataSet trainingDataSet, DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		SVMData svmTrainingData = SVMUtil.getSVMData(trainingDataSet);
		
		svm_problem prob = new svm_problem();
	    prob.y = svmTrainingData.getLabels();
	    prob.l = svmTrainingData.getLabels().length;
	    prob.x = svmTrainingData.getData();
	    
	    svm_parameter parameters = (svm_parameter) additionalData.get(Classifier.SVM_PARAMETERS);
	    this.model = svm.svm_train(prob, parameters);
	}

	/* (non-Javadoc)
	 * @see com.ml.hw6.classifier.Classifier#testModel(com.ml.hw6.data.DataSet, java.util.Map)
	 */
	@Override
	public double testModel(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		SVMData dataSet = SVMUtil.getSVMData(testDataSet);
		double error = 0;
		double[] originalLabels = dataSet.getLabels();
		svm_node[][] testData = dataSet.getData();
		for(int i=0; i < originalLabels.length; i++) {
			double actualLabel = originalLabels[i];
			double predictedLabel = svm.svm_predict(model, testData[i]);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/originalLabels.length;
	}

	/* (non-Javadoc)
	 * @see com.ml.hw6.classifier.Classifier#classifyTestPoint(com.ml.hw6.data.Data, java.util.Map)
	 */
	@Override
	public double classifyTestPoint(Data dataPoint, Map<String, Object> additionalData) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @return the model
	 */
	public svm_model getModel() {
		return model;
	}

	/**
	 * @param model the model to set
	 */
	public void setModel(svm_model model) {
		this.model = model;
	}

}
