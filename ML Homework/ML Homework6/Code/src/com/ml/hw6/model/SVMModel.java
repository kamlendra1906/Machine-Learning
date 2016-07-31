/**
 * 
 */
package com.ml.hw6.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.kami.hw6.svm.SMOImpl;
import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataSet;

/**
 * @author kkumar
 *
 */
public class SVMModel {
	
	private double[] alphas;
	private double b;

	/**
	 * @return the alphas
	 */
	public double[] getAlphas() {
		return alphas;
	}

	/**
	 * @param alphas the alphas to set
	 */
	public void setAlphas(double[] alphas) {
		this.alphas = alphas;
	}

	/**
	 * @return the b
	 */
	public double getB() {
		return b;
	}

	/**
	 * @param b the b to set
	 */
	public void setB(double b) {
		this.b = b;
	}
	
	public double testModel(DataSet trainingData, DataSet testData, Map<String, Object> additionalData) throws Exception {
		double error = 0;
		List<Data> trainingDatas = trainingData.getData();
		List<Data> testDatas = testData.getData();
		
		if(trainingData.dataSize() != alphas.length) {
			throw new Exception("alpha and training data are of different size");
		}
		
		for(Data testPoint : testDatas) {
			double actualLabel = testPoint.labelValue();
			double fxValue = calculateFx(testPoint, trainingDatas, additionalData);
			double predictedLabel = fxValue >= 0 ? 1 : -1;
			//System.out.println(actualLabel + "    "+ predictedLabel);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testDatas.size();
	}

	public double calculateFx(Data testPoint, List<Data> trainingDatas, Map<String, Object> additionalData) throws Exception {
		double value = 0;
		for(int i=0; i < this.alphas.length; i++) {
			Data trainingPoint = trainingDatas.get(i);
			double label = trainingPoint.labelValue();
			if(alphas[i] != 0) {
				value+= alphas[i] * label * SMOImpl.kernel.evaluateKernel(trainingPoint, testPoint, 0, 0, additionalData);
			}
		}
		value+= this.b;
		return value;
	}
	
	public List<Double> calculateFx(List<Data> testPoints, List<Data> trainingPoints, Map<String, Object> additionalData) throws Exception {
		List<Double> fxs = new ArrayList<Double>();
		for(Data testPoint : testPoints) {
			fxs.add(calculateFx(testPoint, trainingPoints, additionalData));
		}
		return fxs;
	}
}