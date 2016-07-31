/**
 * 
 */
package com.ml.hw6.classifier.bagging;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.ml.hw6.classifier.Classifier;
import com.ml.hw6.classifier.impl.AdaBoost;
import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class ECOCImpl implements Classifier {
	
	private int numberOfClasses;
	private int numberOfECOCFunctions;
	private double[][] ecocTable;
	private List<Classifier> adaBoostECOCClassifiers;
	
	public ECOCImpl(int numberOfClasses, int numberOfECOCFunctions) {
		this.numberOfClasses = numberOfClasses;
		this.numberOfECOCFunctions = numberOfECOCFunctions;
		this.adaBoostECOCClassifiers = new ArrayList<Classifier>();
		ecocTable = initializeECOCTable();
	}

	
	@Override
	public void trainModel(DataSet trainingDataSet, DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		List<Double> originalLabels = getOriginalLabel(trainingDataSet);
		for(int function = 0; function < this.numberOfECOCFunctions; function++) {
			
			int featureSize = trainingDataSet.getFeatures().size() - 1;
			
			updateTrainingDataLabelForECOCFunction(trainingDataSet, function);
			
			Classifier adaBoostClassifier = new AdaBoost(featureSize, false, 500);
			adaBoostClassifier.trainModel(trainingDataSet, testDataSet, additionalData);

			adaBoostECOCClassifiers.add(adaBoostClassifier);
			resetTrainingDataLabel(originalLabels, trainingDataSet);
			System.out.println("****************adaboost training done***********************"+ function);
		}
	}

	private List<Double> getOriginalLabel(DataSet trainingDataSet) throws Exception {
		List<Double> originalLabels = new ArrayList<Double>();
		for(Data data : trainingDataSet.getData()) {
			originalLabels.add(data.labelValue());
		}
		return originalLabels;
	}

	private void resetTrainingDataLabel(final List<Double> originalLabels, DataSet trainingDataSet) throws Exception {
		int count=0;
		for(Data data : trainingDataSet.getData()) {
			data.setLabelValue(originalLabels.get(count++));
		}
	}

	private void updateTrainingDataLabelForECOCFunction(DataSet trainingDataSet, int function) throws Exception {
		double[] ecocFunctionClassValues = getClassValuesForECOCFunction(function);
		System.out.println(ClassifierUtil.printArray(ecocFunctionClassValues));
		for(Data data : trainingDataSet.getData()) {
			double oldLabel = data.labelValue();
			double newLabel = getNewLabelForTrainingData(ecocFunctionClassValues, oldLabel);
			data.setLabelValue(newLabel);
		}
	}

	private double[] getClassValuesForECOCFunction(int function) {
		double[] ecocFunctionCLassValues = new double[this.numberOfClasses];
		for(int i = 0; i < this.numberOfClasses; i++) {
			ecocFunctionCLassValues[i] = ecocTable[i][function];
		}
		return ecocFunctionCLassValues;
	}
	
	private double getNewLabelForTrainingData(double[] ecocFunctionClassValues, double labelValue) {
		double newLabel = 0;
		for(int classs=0; classs < this.numberOfClasses; classs++) {
			if(ecocFunctionClassValues[classs] == 1 && classs == labelValue) {
				newLabel = 1;
				break;
			}
		}
		return newLabel;
	}


	/* (non-Javadoc)
	 * @see com.ml.hw4.classifier.Classifier#testModel(com.ml.hw4.data.DataSet, java.util.Map)
	 */
	@Override
	public double testModel(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		double error = 0;
		double[] predictedClassCount = new double[this.numberOfClasses];
		
		for(Data data : testDataSet.getData()) {
			
			double actualClass = data.labelValue();
			double predictedClass = this.classifyTestPoint(data, additionalData);
			predictedClassCount[(int) predictedClass]++;
			
			System.out.println(actualClass + "     "+ predictedClass);
			
			if(actualClass != predictedClass) {
				error++;
			}
		}
		System.out.println(ClassifierUtil.printArray(predictedClassCount));
		return error / testDataSet.dataSize();
	}

	/* (non-Javadoc)
	 * @see com.ml.hw4.classifier.Classifier#classifyTestPoint(com.ml.hw4.data.Data, java.util.Map)
	 */
	@Override
	public double classifyTestPoint(Data dataPoint, Map<String, Object> additionalData) throws Exception {
		double[] predictedBitPattern = new double[this.numberOfECOCFunctions];
		
		for(int function = 0; function < this.numberOfECOCFunctions; function++) {
			Classifier classifier = this.adaBoostECOCClassifiers.get(function);
			predictedBitPattern[function] = classifier.classifyTestPoint(dataPoint, additionalData) == -1 ? 0 : 1;
		}
		
		System.out.println(ClassifierUtil.printArray(predictedBitPattern));
		return predictClass(predictedBitPattern);
	}

	private double predictClass(double[] predictedBitPattern) {
		double minHummingDistance = Double.POSITIVE_INFINITY;
		double mostLikelyClass = Double.NaN;
		
		for(int classs=0; classs < this.numberOfClasses; classs++) {
			double[] classBitPattern = this.ecocTable[classs];
			double hummingDistance = getHummingDistance(classBitPattern, predictedBitPattern);
			if(hummingDistance < minHummingDistance) {
				minHummingDistance = hummingDistance;
				mostLikelyClass = classs;
			}
		}
		return mostLikelyClass;
	}

	private double getHummingDistance(double[] classBitPattern, double[] predictedBitPattern) {
		double hummingDistance = 0;
		for(int i=0; i < this.numberOfECOCFunctions; i++) {
			if(classBitPattern[i] != predictedBitPattern[i]) {
				hummingDistance++;
			}
		}
		return hummingDistance;
	}
	
	private double[][] initializeECOCTable() {
		double[][] ecocTable = new double[numberOfClasses][numberOfECOCFunctions];
		List<double[]> ecocFunctions = getECOCFunctions();
		for(int classs = 0; classs < numberOfClasses; classs++) {
			for(int function = 0; function < numberOfECOCFunctions; function++) {
				ecocTable[classs][function] = ecocFunctions.get(function)[classs];
			}
		}
		return ecocTable;
	}
	
	@SuppressWarnings("unused")
	private double[][] initializeECOCTableUniform() {
		double[][] ecocTable = new double[numberOfClasses][numberOfECOCFunctions];
		for(int i=0; i< numberOfClasses ; i++) {
			for(int j=0; j< numberOfECOCFunctions; j++) {
				if(i == j) {
					ecocTable[i][j] = 1;
				}
			}
		}
		return ecocTable;
	}

	private List<double[]> getECOCFunctions() {
		Random random = new Random();
		Set<Integer> alreadySelectedECOCFunctionValue = new HashSet<Integer>();
		List<double[]> ecocFunctions = new ArrayList<double[]>();
		
		while(ecocFunctions.size() != numberOfECOCFunctions) {
			double[] ecocFunction = new double[numberOfClasses];
			int ecocFunctionValue = 0;
			int numberOfOnesInFunction = 0;
			
			for(int classs = 0; classs < numberOfClasses; classs++) {
				
				double randomValue = random.nextDouble();
				double bitValue = randomValue > 0.5d ? 1 : 0;
				
				if(bitValue == 1) {
					numberOfOnesInFunction++;
					ecocFunctionValue += Math.pow(2, classs);
				}
				
				ecocFunction[classs] = bitValue;
			}
			if(numberOfOnesInFunction >= 3 && !alreadySelectedECOCFunctionValue.contains(ecocFunctionValue)) {
				ecocFunctions.add(ecocFunction);
				alreadySelectedECOCFunctionValue.add(ecocFunctionValue);
			}
		}
			
		return ecocFunctions;
	}

	@SuppressWarnings("unused")
	private void printECOCTable() {
		for(int i=0; i < this.numberOfClasses; i++) {
			System.out.println(ClassifierUtil.printArray(this.ecocTable[i]));
		}
	}

}