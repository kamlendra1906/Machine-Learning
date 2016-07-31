/**
 * 
 */
package src.com.ml.hw3.classifier.stats;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class GaussianModel {

	private int featureDimension;
	private Matrix mean;
	private Matrix covariance;
	
	public GaussianModel(int featureDimension) {
		this.featureDimension = featureDimension;
		this.mean = Matrix.random(featureDimension, 1).times(10);
		this.covariance = Matrix.identity(featureDimension, featureDimension);
	}

	/**
	 * @return the featureDimension
	 */
	public int getFeatureDimension() {
		return featureDimension;
	}

	/**
	 * @param featureDimension the featureDimension to set
	 */
	public void setFeatureDimension(int featureDimension) {
		this.featureDimension = featureDimension;
	}

	/**
	 * @return the mean
	 */
	public Matrix getMean() {
		return mean;
	}

	/**
	 * @param mean the mean to set
	 */
	public void setMean(Matrix mean) {
		this.mean = mean;
	}

	/**
	 * @return the covariance
	 */
	public Matrix getCovariance() {
		return covariance;
	}

	/**
	 * @param covariance the covariance to set
	 */
	public void setCovariance(Matrix covariance) {
		this.covariance = covariance;
	}
	
	public double getGenerativeProbabilityOfPoint(Matrix x) {
		double exponentPart = calculateExponentPart(x);
		return Math.pow(Math.PI * 2, (double)-this.featureDimension/2) * Math.pow(this.covariance.det(), -0.5) * exponentPart;
	}

	private double calculateExponentPart(Matrix x) {
		Matrix xMinusMue = x.minus(this.mean);
		double weirdMatrixCalculation = xMinusMue.transpose().times(this.covariance.inverse()).times(xMinusMue).det();
		weirdMatrixCalculation/= 2;
		return Math.exp(-weirdMatrixCalculation);
	}
}