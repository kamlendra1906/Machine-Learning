package com.ml.hw2.classifier;

import com.ml.hw2.data.DataForRegression;

import Jama.Matrix;

public class RidgeRegressionJama {
	
	public static Matrix train(final DataForRegression trainingData, final double lambda) throws	Exception {
		Matrix x = new Matrix(trainingData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(trainingData.getValueData(), trainingData.getSampleSize());
		Matrix xTranspose = x.transpose();
		Matrix xTransposeMultX = xTranspose.times(x);
		Matrix identity = Matrix.identity(xTransposeMultX.getRowDimension(), xTransposeMultX.getColumnDimension());
		identity = identity.times(lambda);
		Matrix xTransposeXPlusLambdaI = xTransposeMultX.plus(identity);
		Matrix dagger = xTransposeXPlusLambdaI.inverse();
		Matrix daggerMult = dagger.times(xTranspose);
		Matrix weight = daggerMult.times(y);
		return weight;
	}
}