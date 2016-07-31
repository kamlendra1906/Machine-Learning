/**
 * 
 */
package com.kami.hw7.perceptron;

import java.util.List;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;

/**
 * @author kkumar
 *
 */
public class DualPerceptronImpl {

	private SVMKernel kernal;
	double[] alphas;
	
	public DualPerceptronImpl(SVMKernel kernel, int dataSize) {
		this.kernal = kernel;
		alphas = new double[dataSize];
	}
	
	public void train(DataSet trainingData, Map<String, Object> additionalData) throws Exception {
		int iteration = 0;
		while (true) {
			double changedAlphas = runIteration(trainingData, additionalData);
			System.out.println("Iteration"+ iteration+++"  Changed alphas: "+changedAlphas+"  Error: "+changedAlphas/trainingData.dataSize());
			if(changedAlphas == 0) {
				break;
			}
		}
	}

	private double runIteration(DataSet trainingData, Map<String, Object> additionalData) throws Exception {
		double changeCount = 0;
		List<Data> trainingPoints = trainingData.getData();
		for(int i=0; i < trainingPoints.size(); i++) {
			Data dataPoint = trainingPoints.get(i);
			double actualLabel = dataPoint.labelValue();
			double predictedLabel = predictLabel(dataPoint, trainingPoints, additionalData);
			if(actualLabel * predictedLabel <= 0) {
				alphas[i]+= dataPoint.labelValue();
				changeCount++;
			}
		}
		return changeCount;
	}

	private double predictLabel(Data dataPoint, List<Data> trainingPoints, Map<String, Object> additionalData) throws Exception {
		double result = 0;
		for(int i=0; i< trainingPoints.size(); i++) {
			result+= alphas[i] * kernal.evaluateKernel(dataPoint, trainingPoints.get(i), 0, 0, additionalData);
		}
		return Math.signum(result);
	}
	
	public double testModel(DataSet testData, DataSet trainingData, Map<String, Object> additionalData) throws Exception {
		double error = 0;
		for(Data data : testData.getData()) {
			double actualLabel = data.labelValue();
			double predictedLabel = predictLabel(data, trainingData.getData(), additionalData);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testData.dataSize();
	}
}
