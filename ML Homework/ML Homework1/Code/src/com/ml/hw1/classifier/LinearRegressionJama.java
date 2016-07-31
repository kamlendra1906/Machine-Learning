package com.ml.hw1.classifier;

import com.ml.hw1.data.DataForRegression;

import Jama.Matrix;

public class LinearRegressionJama {
	
	public static Matrix train(DataForRegression trainingData) throws	Exception {
		Matrix x = new Matrix(trainingData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(trainingData.getValueData(), trainingData.getSampleSize());
		Matrix xTranspose = x.transpose();
		Matrix xTransposeMultX = xTranspose.times(x);
		Matrix dagger = xTransposeMultX.inverse();
		Matrix daggerMult = dagger.times(xTranspose);
		Matrix weight = daggerMult.times(y);
		return weight;
	}
	
	public static double test(DataForRegression testData, Matrix weight) throws Exception {
		Matrix x = new Matrix(testData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(testData.getValueData(), testData.getSampleSize());
		Matrix predictedX = x.times(weight);
		Matrix errorMatrix = predictedX.minus(y);
		int rows = errorMatrix.getRowDimension();
		int columns = errorMatrix.getColumnDimension();
		double squaredError = 0;
		for(int i=0; i<rows;i++) {
			for(int j=0;j<columns; j++) {
				squaredError+= Math.pow(errorMatrix.getArray()[i][j], 2);
			}
		}
		return squaredError/rows;	
	}
}