/**
 * 
 */
package com.kami.hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
public class Q3 {
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		System.out.println("Reading input file");
		DataSet trainingData = DataInput.getDigitData(ClassifierUtil.DIGIT_TRAINING_DATA_FILE_MODIFIED, 400);
		
		System.out.println("reading output file");
		DataSet testData = DataInput.getDigitData(ClassifierUtil.DIGIT_TEST_DATA_FILE_MODFIED, 400);
		
		System.out.println("sampling training data");
		trainingData.setData(sampleTrainingData(trainingData.getData(), 3));
		
		System.out.println("Normalizing data");
		DataInput.normalizeData(trainingData, testData);
		
		Collections.shuffle(trainingData.getData());
		
		Map<String, Object> parameters = getAdditionalData();
		Map<Integer, SVMModel> labelClassifierMap = new HashMap<Integer, SVMModel>();
		for(int label : trainingData.getClasses()) {
			List<Double> originalLabels = getOriginalLabels(trainingData.getData());
			updateLabelForClass(label, trainingData);
			SMOImpl smo = new SMOImpl(trainingData.dataSize());
			labelClassifierMap.put(label, smo.train(trainingData, parameters));
			resetLabels(trainingData.getData(), originalLabels);
			System.out.println("trained for label:" + label);
		}
		
		double error = testModel(testData, trainingData, labelClassifierMap, parameters);
		System.out.println(error);
	}

	private static double testModel(DataSet testData, DataSet trainingData, Map<Integer, SVMModel> labelClassifierMap,
			Map<String, Object> parameters) throws Exception {
		parameters.put(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED, false);
		parameters.put(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED, false);
		double error = 0;
		Map<Integer, List<Double>> labelFXMap = getLabelFxMap(testData, trainingData, labelClassifierMap, parameters);
		List<Data> testPoints = testData.getData();
		for(int i=0; i < testPoints.size(); i++) {
			Data testPoint = testPoints.get(i);
			double actualLabel = testPoint.labelValue();
			double predictedLabel = predictLabel(i, labelFXMap);
			System.out.println(actualLabel+"   "+predictedLabel);
			if(predictedLabel != actualLabel) {
				error++;
			}
		}
		return error/testData.getData().size();
	}

	private static double predictLabel(int i, Map<Integer, List<Double>> labelFXMap) {
		double maxFx = Double.NEGATIVE_INFINITY;
		double maxFxLabel = Double.NaN;
		
		for(Entry<Integer, List<Double>> entry : labelFXMap.entrySet()) {
			int label = entry.getKey();
			List<Double> fxs = entry.getValue();
			
			if(fxs.get(i) > maxFx) {
				maxFx = fxs.get(i);
				maxFxLabel = label;
			}
		}
		return maxFxLabel;
	}

	private static Map<Integer, List<Double>> getLabelFxMap(DataSet testData, DataSet trainingData, Map<Integer, SVMModel> labelClassifierMap, 
			Map<String, Object> parameters) throws Exception {
		
		Map<Integer, List<Double>> labelFxMap = new HashMap<Integer, List<Double>>();
		
		for(Entry<Integer, SVMModel> entry : labelClassifierMap.entrySet()) {
			int label = entry.getKey();
			SVMModel model = entry.getValue();
			
			List<Double> originalLabels = getOriginalLabels(trainingData.getData());
			updateLabelForClass(label, trainingData);
			labelFxMap.put(label, model.calculateFx(testData.getData(), trainingData.getData(), parameters));
			resetLabels(trainingData.getData(), originalLabels);
		}
		
		return labelFxMap;
	}

	private static void updateLabelForClass(int label, DataSet trainingData) throws Exception {
		for(Data data : trainingData.getData()) {
			if(data.labelValue() != label) {
				data.setLabelValue(-1);
			} else {
				data.setLabelValue(1);
			}
		}
	}

	private static List<Data> sampleTrainingData(List<Data> trainingData, double samplePercent) throws Exception {
		List<Data> result = new ArrayList<Data>();
		@SuppressWarnings("unchecked")
		List<Data>[] imageListByClass = new ArrayList[10];
		
		for(Data img : trainingData) {
			List<Data> list = imageListByClass[(int)img.labelValue()];
			if(list == null) {
				list = new ArrayList<Data>();
				imageListByClass[(int)img.labelValue()] = list;
			}
			list.add(img);
		}
		
		for(List<Data> images : imageListByClass) {
			Collections.shuffle(images);
			sample(images, result, samplePercent);
		}
		return result;
	}
	
	private static void sample(List<Data> images, List<Data> result, double samplePercent) {
		int numOfDataToBeAdded = (int) ((images.size() * samplePercent) /100);
		for(int i=0; i < numOfDataToBeAdded; i++) {
			result.add(images.get(i));
		}
	}
	
	private static Map<String, Object> getAdditionalData() {
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
		parameters.put(SVMUtil.SVM_PARAMETER_NU, 2.0);
		return parameters;
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
}
