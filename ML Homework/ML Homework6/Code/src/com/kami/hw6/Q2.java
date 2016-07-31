/**
 * 
 */
package com.kami.hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.kami.hw6.svm.SMOImpl;
import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataInput;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.model.SVMModel;
import com.ml.hw6.util.ClassifierUtil;
import com.ml.hw6.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class Q2 {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAMBASE_ORIGINAL_TRAINING_DATA_FILE, ClassifierUtil.SPAMBASE_ORIGINAL_FEATURE_FILE);
		
		int totalFolds = 10;
		int dataPerFold = dataSet.dataSize() / totalFolds;
		double[] avgError = new double[2];
		
		Collections.shuffle(dataSet.getData(), new Random(100));
		
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
			List<Double> originalTrainingLabels = getOriginalLabels(trainingData.getData());
			List<Double> originalTestLabels = getOriginalLabels(testData.getData());
			updateLabel(trainingData.getData());
			updateLabel(testData.getData());
			
			DataInput.normalizeData(trainingData, testData);
			
			Map<String, Object> parameters = getParameters();
			SVMModel model = new SMOImpl(trainingData.dataSize()).train(trainingData, parameters);
			
			parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
			parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
			
			double trainingError = model.testModel(trainingData, trainingData, parameters);
			double testError = model.testModel(trainingData, testData, parameters);
			avgError[0]+= trainingError;
			avgError[1]+= testError;
			resetLabels(trainingData.getData(), originalTrainingLabels);
			resetLabels(testData.getData(), originalTestLabels);
			break;
		}
		System.out.println(ClassifierUtil.printArray(avgError));
	}
	
	private static List<Double> getOriginalLabels(List<Data> datas) throws Exception {
		List<Double> originalLabels = new ArrayList<Double>();
		for(Data data : datas) {
			originalLabels.add(data.labelValue());
		}
		return originalLabels;
	}
	
	private static void resetLabels(List<Data> datas, List<Double> labels) throws Exception {
		for(int i=0; i < datas.size(); i++) {
			datas.get(i).setLabelValue(labels.get(i));
		}
	}

	private static void updateLabel(List<Data> sampledData) throws Exception {
		for(Data data : sampledData) {
			if(data.labelValue() == 0) {
				data.setLabelValue(-1);
			}
		}
	}

	private static Map<String, Object> getParameters() {
		Map<String, Object> parameters = new HashMap<String, Object>();
		parameters.put(SVMUtil.SVM_PARAMETER_C, .75d);
		parameters.put(SVMUtil.SVM_PARAMETER_TOLERANCE, .0001d);
		parameters.put(SVMUtil.SVM_PARAMETER_PASSES, 20);
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, true);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, true);
		
		parameters.put(SVMUtil.SVM_PARAMETER_ALPHA, .1d);
		parameters.put(SVMUtil.SVM_PARAMETER_BETA, 1d);
		parameters.put(SVMUtil.SVM_PARAMETER_DEGREE, 3d);
		parameters.put(SVMUtil.SVM_PARAMETER_KERNEL_TYPE, SVMUtil.RBF_KERNEL);
		parameters.put(SVMUtil.SVM_PARAMETER_NU, .5);
		return parameters;
	}
	
	private static List<Data> sampleData(List<Data> data, double percent) {
		List<Data> result = new ArrayList<Data>();
		int size = (int) (data.size() * percent / 100);
		int counter = 0;
		while(result.size() < size) {
			result.add(data.get(counter++));
		}
		return result;
	}
}
