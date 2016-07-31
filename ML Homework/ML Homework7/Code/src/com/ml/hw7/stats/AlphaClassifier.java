/**
 * 
 */
package com.ml.hw7.stats;

import com.ml.hw7.classifier.Classifier;

/**
 * @author kkumar
 *
 */
public class AlphaClassifier {

	private double alpha;
	private Classifier classifier;
	
	public AlphaClassifier(double alpha, Classifier classifier) {
		this.alpha = alpha;
		this.classifier = classifier;
	}

	/**
	 * @return the alpha
	 */
	public double getAlpha() {
		return alpha;
	}

	/**
	 * @param alpha the alpha to set
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * @return the classifier
	 */
	public Classifier getClassifier() {
		return classifier;
	}

	/**
	 * @param classifier the classifier to set
	 */
	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}	
}