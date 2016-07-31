/**
 * 
 */
package com.ml.hw5.classifier.bagging;

import com.ml.hw5.data.DataForRegression;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class L2BatchGradientDescentImpl {

	private double learningRate;
	private double regularizationFactor;
	private int featureSize;
	private Matrix weightValue;
	
	public L2BatchGradientDescentImpl(double learningRate, int featureSize, double regularizationFactor) {
		this.learningRate = learningRate;
		this.featureSize = featureSize;
		this.regularizationFactor = regularizationFactor;
		weightValue = Matrix.random(featureSize,1);
	}
	
	public Matrix findOptimalWeight(DataForRegression trainingData, boolean linearRegression) throws Exception {
		if(trainingData == null) {
			throw new Exception("training data is null");
		}
		int rows = trainingData.getSampleSize();
		int cols = trainingData.getFeatureSize();
		
		Matrix trainingFeatureData = new Matrix(trainingData.getTwoDArrayFeatureData(), rows, cols);
		Matrix trainingLabelvalue = new Matrix(trainingData.getValueData(), rows);
		
		int dataPass = 0;
		while(true) {
			double[] roundWeightUpdate = new double[featureSize];
			
			for(int row = 0; row < rows; row++) {
				double predictedValue = 0;
				
				for(int col=0; col < cols; col++) {
					predictedValue+= weightValue.get(col,0) * trainingFeatureData.get(row, col);
				}
				
				if(!linearRegression) {
					predictedValue = getLogisticRegressionValue(predictedValue);
				}
				double error = predictedValue - trainingLabelvalue.get(row, 0);
				
				for(int col=0; col < cols; col++) {
					roundWeightUpdate[col]+= error * trainingFeatureData.get(row, col);
				}
			}
			
			for(int col=0; col < cols; col++) {
				double newWeightValue = getNewFeatureWeight(weightValue.get(col,0), roundWeightUpdate, col, rows);
				weightValue.set(col, 0, newWeightValue);
			}
			if(dataPass == 2500) {
				return weightValue;
			}
			dataPass++;
			System.out.println(dataPass);
		}
	}

	private double getNewFeatureWeight(double wOld, double[] roundWeightUpdate , int col, int rows) {
		double value = roundWeightUpdate[col];
		if(col == 0) {
			value = value * learningRate/rows;
			return wOld - value;
		}
		value/= rows;
		value+= wOld * regularizationFactor / rows;
		value*= learningRate;
		return wOld - value;
	}

	private double getLogisticRegressionValue(double predictedValue) {
		return 1/(1+ Math.exp(-predictedValue));
	}
}