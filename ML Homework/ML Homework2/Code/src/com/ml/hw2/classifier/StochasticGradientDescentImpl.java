/**
 * 
 */
package com.ml.hw2.classifier;

import com.ml.hw2.data.DataForRegression;
import com.ml.hw2.util.ClassifierUtil;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class StochasticGradientDescentImpl {

	private double lambda;
	private Matrix weightValue;
	
	public StochasticGradientDescentImpl(double lambda, int featureSize) {
		this.lambda = lambda;
		weightValue = Matrix.random(featureSize,1);
	}
	
	public Matrix findOptimalWeight(DataForRegression trainingData, double threshold ,
			boolean linearRegression) throws Exception {
		if(trainingData == null) {
			throw new Exception("training data is null");
		}
		
		Matrix trainingFeatureData = new Matrix(trainingData.getTwoDArrayFeatureData(), trainingData.getSampleSize(), trainingData.getFeatureSize());
		Matrix trainingLabelvalue = new Matrix(trainingData.getValueData(), trainingData.getSampleSize());
		
		double previousError = ClassifierUtil.testWeight(trainingData, weightValue, linearRegression);
		int dataPass = 0;
		while(true) {
			double currentError = 0;
			for(int row = 0; row < trainingFeatureData.getRowDimension(); row++) {
				double predictedValue = 0;
				for(int col=0; col < trainingFeatureData.getColumnDimension(); col++) {
					predictedValue+= weightValue.get(col,0) * trainingFeatureData.get(row, col);
				}
				if(!linearRegression) {
					predictedValue = getLogisticRegressionValue(predictedValue);
				}
				for(int col=0; col < trainingFeatureData.getColumnDimension(); col++) {
					double newWeightValue = getNewFeatureWeight(weightValue.get(col,0), predictedValue, trainingLabelvalue.get(row, 0), trainingFeatureData.get(row, col), linearRegression);
					weightValue.set(col, 0, newWeightValue);
				}
				//System.out.println("pass= "+(dataPass)+" row="+row+"   weight=  "+printArray(weightValue.getRowPackedCopy()));
			}
			currentError = ClassifierUtil.testWeight(trainingData, weightValue, linearRegression);
			/*if(converged(previousError, currentError, threshold)) {
				return weightValue;
			}*/
			//System.out.println("Scan = "+(dataPass)+ " changeInError="+ (Math.abs(previousError-currentError)));
			previousError = currentError;
			if(dataPass == 10000) {
				return weightValue;
			}
			dataPass++;
		}
	}

	
	private boolean converged(double previousError, double currentError, double threshold) {
		return Math.abs(previousError-currentError) < threshold;
	}

	private double getNewFeatureWeight(double wOld, double predictedValue, double actualValue, double featureValue, boolean linearRegression) {
		double wNew = 0;
		if(linearRegression) {
			wNew = wOld - lambda * (predictedValue - actualValue) * featureValue;
		} else {
			wNew = wOld + lambda * (actualValue - predictedValue) * featureValue;
		}
		return wNew;
	}

	private double getLogisticRegressionValue(double predictedValue) {
		return 1/(1+ Math.exp(-predictedValue));
	}	
}
