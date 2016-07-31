/**
 * 
 */
package com.ml.hw2.classifier;

import com.ml.hw2.data.DataForRegression;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class PerceptronClassifierImpl {

	private double lambda;
	private Matrix weight;

	public PerceptronClassifierImpl(double lambda, int featureSize) {
		this.lambda = lambda;
		weight = Matrix.random(featureSize, 1);
		//weight = new Matrix(featureSize, 1);
	}

	public Matrix train(DataForRegression trainingData) throws Exception {
		if (trainingData == null) {
			throw new Exception("dataset is null");
		}

		Matrix trainingFeatureData = new Matrix(trainingData.getTwoDArrayFeatureData(), trainingData.getSampleSize(),
				trainingData.getFeatureSize());
		Matrix trainingLabelvalue = new Matrix(trainingData.getValueData(), trainingData.getSampleSize());

		int dataPass = 0;
		while (true) {
			int errorCount = 0;
			for (int row = 0; row < trainingFeatureData.getRowDimension(); row++) {
				double predictedValue = 0;
				double actualValue = trainingLabelvalue.get(row, 0);
				for (int col = 0; col < trainingFeatureData.getColumnDimension(); col++) {
					predictedValue += weight.get(col, 0) * trainingFeatureData.get(row, col);
				}
				if (predictedValue * actualValue <= 0) {
					errorCount++;
					for (int col = 0; col < trainingFeatureData.getColumnDimension(); col++) {
						weight.set(col, 0, getNewWeight(weight.get(col, 0), trainingFeatureData.get(row, col), actualValue));
					}
				}
			}
			System.out.println("Scan= "+dataPass+ "  total_mistake= "+errorCount);
			if(errorCount == 0) {
				return weight;
			}
			dataPass++;
		}
	}

	private double getNewWeight(double wOld, double featureValue, double actualValue) {
		return wOld + lambda * featureValue * actualValue;
	}
}