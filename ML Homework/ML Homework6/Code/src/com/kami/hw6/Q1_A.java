/**
 * 
 */
package com.kami.hw6;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.ml.hw6.classifier.Classifier;
import com.ml.hw6.classifier.impl.SVMImpl;
import com.ml.hw6.data.DataInput;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.util.ClassifierUtil;

import libsvm.svm_parameter;

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
		
		int totalFolds = 10;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		double[] avgError = new double[2];
		
		Collections.shuffle(dataSet.getData());
		
		for (int fold = 0; fold < totalFolds; fold++) {
		
			DataSet trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			
			for (int counter = 0; counter < dataSet.dataSize(); counter++) {
				if (counter >= fold * dataPerFold && counter < (fold + 1) * dataPerFold) {
					testData.addData(dataSet.getData().get(counter));
				} else {
					trainingData.addData(dataSet.getData().get(counter));
				}
			}
			DataInput.normalizeData(trainingData, testData);
			double[] errors = runSVM(trainingData, testData);
			avgError[0]+= errors[0];
			avgError[1]+= errors[1];
			System.out.println(ClassifierUtil.printArray(errors));
		}
		ClassifierUtil.normalizeProbability(avgError, 10);
		System.out.println("Average error");
		System.out.println(ClassifierUtil.printArray(avgError));
	}

	private static double[] runSVM(DataSet trainingData, DataSet testData) throws Exception {
		double[] errors = new double[2];
		
		SVMImpl svm = new SVMImpl();
		svm.trainModel(trainingData, testData, getAdditionalDataMap());

		errors[0] = svm.testModel(trainingData, null);
	    errors[1] = svm.testModel(testData, null);
	    return errors;
	}
	
	private static Map<String, Object> getAdditionalDataMap() {
		Map<String, Object> additionalDataMap = new HashMap<String, Object>();
		additionalDataMap.put(Classifier.SVM_PARAMETERS, getSVMLinearKernalParameters());
		return additionalDataMap;
	}
	
	@SuppressWarnings("unused")
	private static svm_parameter getSVMLinearKernalParameters() {
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.LINEAR;
		param.degree = 1;
		param.gamma = 0;
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 50;
		param.eps = 0.1;
		param.p = 0.1;
		param.shrinking = 0;
		param.probability = 0;
		param.nr_weight = 0;
	    return param;
	}
	
	@SuppressWarnings("unused")
	private static svm_parameter getSVMPolyKernalParameters() {
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.POLY;
		param.degree = 2;
		param.gamma = 0.1;
		param.coef0 = 1;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 50;
		param.eps = 0.1;
		param.p = 0.1;
		param.shrinking = 0;
		param.probability = 0;
		param.nr_weight = 0;
	    return param;
	}
	
	private static svm_parameter getSVMRBFKernalParameters() {
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 2;
		param.gamma = 0.1;
		param.coef0 = 1;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 50;
		param.eps = 0.1;
		param.p = 0.1;
		param.shrinking = 0;
		param.probability = 0;
		param.nr_weight = 0;
	    return param;
	}
}