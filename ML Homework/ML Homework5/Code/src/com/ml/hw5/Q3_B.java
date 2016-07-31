/**
 * 
 */
package com.ml.hw5;

import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Classifier;
import mltk.predictor.Learner.Task;
import mltk.predictor.evaluation.Evaluator;
import mltk.predictor.glm.LassoLearner;

/**
 * @author kkumar
 *
 */
public class Q3_B {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		/*DataSet trainingData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_DATA_FILE,
				ClassifierUtil.SPAMBASE_POLLUTED_TRAINING_LABEL_FILE);
		DataSet testData = DataInput.getDataForHW5(ClassifierUtil.SPAMBASE_POLLUTED_TEST_DATA_FILE,
				ClassifierUtil.SPAMBASE_POLLUTED_TEST_LABEL_FILE);
		DataInput.normalizeData(trainingData, testData);
		
		ClassifierUtil.writeDataToFile(
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\mltptraining.txt", trainingData);
		ClassifierUtil.writeDataToFile(
				"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\mltptest.txt", trainingData);*/
	
		Instances trainingInstances = InstancesReader.read("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\mltptraining.txt", 1057);
		Instances testInstances = InstancesReader.read("C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\mltptest.txt", 1057);
		Classifier learner = train(trainingInstances);
		System.out.println(Evaluator.evalError(learner, testInstances));
		System.out.println("done");
	}
	
	public static Classifier train(Instances instances) throws Exception {
		LassoLearner learner = new LassoLearner();
		learner.setVerbose(true);
		learner.setTask(Task.CLASSIFICATION);
		learner.setLambda(0.1);
		learner.setMaxNumIters(100);
		return learner.build(instances);
	}
 
}
