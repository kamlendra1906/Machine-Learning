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
import com.ml.hw5.classifier.bagging.ECOCImpl;
import com.ml.hw5.classifier.impl.AdaBoost;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.data.Image;
import com.ml.hw5.data.ImageDataInput;
import com.ml.hw5.data.MNISTReader;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class Q5 {

	/**
	 * @param args
	 * @throws Exception 
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
		
		System.out.println("*******************************runing ecoc********************************************");
		Map<String, Object> additionalData = getAdditionalData();
		Classifier classifier = new ECOCImpl(10, 50);
		classifier.trainModel(trainingData, testData, additionalData);
		System.out.println("training done!");
		double error = classifier.testModel(testData, additionalData);
		System.out.println(error);
		
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
	
	private static Map<String, Object> getAdditionalData() {
		Map<String, Object> additionalData = new HashMap<String, Object>();
		additionalData.put(AdaBoost.GENERATE_ROUND_STATS, false);
		additionalData.put(AdaBoost.ALL_CONFUSION_MATRIX, new ArrayList<double[]>());
		additionalData.put(AdaBoost.GENERATE_ACTIVE_LEARNING_STATS, false);
		additionalData.put(AdaBoost.CLASSIFICATION_THRESHOLD, 0d);
		additionalData.put(AdaBoost.GENERATE_CONFUSION_MATRIX, false);
		return additionalData;
	}
}