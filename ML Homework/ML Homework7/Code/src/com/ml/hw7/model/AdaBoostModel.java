package com.ml.hw7.model;

import java.util.ArrayList;
import java.util.List;

import com.ml.hw7.classifier.Classifier;

public class AdaBoostModel implements Model {

	private List<Double> classifierWeight;
	private List<Classifier> classifiers;
	
	public AdaBoostModel() {
		classifierWeight = new ArrayList<Double>();
		classifiers = new ArrayList<Classifier>();
	}

	/**
	 * @return the classifierWeight
	 */
	public List<Double> getClassifierWeight() {
		return classifierWeight;
	}

	/**
	 * @param classifierWeight the classifierWeight to set
	 */
	public void setClassifierWeight(List<Double> classifierWeight) {
		this.classifierWeight = classifierWeight;
	}

	/**
	 * @return the classifiers
	 */
	public List<Classifier> getClassifiers() {
		return classifiers;
	}

	/**
	 * @param classifiers the classifiers to set
	 */
	public void setClassifiers(List<Classifier> classifiers) {
		this.classifiers = classifiers;
	}
}