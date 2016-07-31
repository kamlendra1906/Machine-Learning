/**
 * 
 */
package com.kami.hw6;
import java.util.ArrayList;
import java.util.List;

import com.ml.hw6.data.DataSet;
import com.ml.hw6.data.Image;
import com.ml.hw6.data.ImageDataInput;
import com.ml.hw6.data.MNISTReader;
import com.ml.hw6.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class DataMover {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		ImageDataInput featureExtractor = new ImageDataInput(200);
		System.out.println("*******************************reading training file********************************************");
		List<Image> trainingImageData = MNISTReader.getImages(ClassifierUtil.DIGIT_TRAINING_LABEL_FILE, ClassifierUtil.DIGIT_TRAINING_DATA_FILE);	
		
		/*System.out.println("*******************************reading test file********************************************");
		
		
		System.out.println("*******************************sampling training data********************************************");
		//List<Image> sampledTrainingImageData = sampleTrainingData(trainingImageData, 10);
		
		
		
		System.out.println("*******************************extracting training features********************************************");
		DataSet trainingData = featureExtractor.getImageDataSet(trainingImageData);
		
		System.out.println("*******************************extracting test features********************************************");
		DataSet testData = featureExtractor.getImageDataSet(testImageData);
		
		ClassifierUtil.writeDataToFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework6\\Data\\Digit\\Training\\digit-training.data", trainingData);
		ClassifierUtil.writeDataToFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework6\\Data\\Digit\\Training\\digit-test.data", testData);*/
		
		writeTrainingData(featureExtractor, trainingImageData);
		List<Image> testImageData = MNISTReader.getImages(ClassifierUtil.DIGIT_TEST_LABEL_FILE, ClassifierUtil.DIGIT_TEST_DATA_FILE);
		writeTestData(featureExtractor, testImageData);
	}
	
	private static void writeTrainingData(ImageDataInput featureExtractor, List<Image> testImageData) throws Exception {
		int totalFolds = 10;
		int dataPerFold = testImageData.size() / totalFolds;
		
		for (int fold = 0; fold < totalFolds; fold++) {
			List<Image> images = new ArrayList<Image>();
			
			for (int counter = 0; counter < testImageData.size(); counter++) {
				if (counter >= fold * dataPerFold && counter < (fold + 1) * dataPerFold) {
					images.add(testImageData.get(counter));
				}
			}
			DataSet testData = featureExtractor.getImageDataSet(testImageData);
			ClassifierUtil.writeDataToFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework6\\Data\\Digit\\Training\\digit-test"+fold+".data", testData);
		}
	}

	public static void writeTestData(ImageDataInput featureExtractor, List<Image> trainingImageData) throws Exception {
		DataSet trainingData = featureExtractor.getImageDataSet(trainingImageData);
		ClassifierUtil.writeDataToFile("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework6\\Data\\Digit\\Training\\digit-training.data", trainingData);
	}

}
