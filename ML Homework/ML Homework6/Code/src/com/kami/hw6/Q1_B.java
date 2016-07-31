/**
 * 
 */
package com.kami.hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ml.hw6.classifier.Classifier;
import com.ml.hw6.classifier.impl.SVMImpl;
import com.ml.hw6.data.DataInput;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.data.Image;
import com.ml.hw6.data.ImageDataInput;
import com.ml.hw6.data.MNISTReader;
import com.ml.hw6.util.ClassifierUtil;

import libsvm.svm_parameter;

/**
 * @author kkumar
 *
 */
public class Q1_B {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("*******************************reading training file********************************************");
		List<Image> trainingImageData = MNISTReader.getImages(ClassifierUtil.DIGIT_TRAINING_LABEL_FILE, ClassifierUtil.DIGIT_TRAINING_DATA_FILE);	
		
		System.out.println("*******************************reading test file********************************************");
		List<Image> testImageData = MNISTReader.getImages(ClassifierUtil.DIGIT_TEST_LABEL_FILE, ClassifierUtil.DIGIT_TEST_DATA_FILE);
		
		System.out.println("*******************************sampling training data********************************************");
		List<Image> sampledTrainingImageData = sampleTrainingData(trainingImageData, 20);
		
		ImageDataInput featureExtractor = new ImageDataInput(200);
		
		System.out.println("*******************************extracting training features********************************************");
		DataSet trainingData = featureExtractor.getImageDataSet(sampledTrainingImageData);
		
		System.out.println("*******************************extracting test features********************************************");
		DataSet testData = featureExtractor.getImageDataSet(testImageData);
		
		System.out.println("*******************************runing SVM********************************************");
		
		DataInput.normalizeData(trainingData, testData);
		double[] errors = runSVM(trainingData, testData);
		
		System.out.println(ClassifierUtil.printArray(errors));
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

	
	
	private static List<Image> sampleTrainingData(List<Image> trainingData, double samplePercent) {
		List<Image> result = new ArrayList<Image>();
		@SuppressWarnings("unchecked")
		List<Image>[] imageListByClass = new ArrayList[10];
		
		for(Image img : trainingData) {
			List<Image> list = imageListByClass[(int)img.getLabel()];
			if(list == null) {
				list = new ArrayList<Image>();
				imageListByClass[(int)img.getLabel()] = list;
			}
			list.add(img);
		}
		
		for(List<Image> images : imageListByClass) {
			Collections.shuffle(images);
			sample(images, result, samplePercent);
		}
		return result;
	}

	private static void sample(List<Image> images, List<Image> result, double samplePercent) {
		int numOfDataToBeAdded = (int) ((images.size() * samplePercent) /100);
		for(int i=0; i < numOfDataToBeAdded; i++) {
			result.add(images.get(i));
		}
	}
}