/**
 * 
 */
package com.kami.hw7.svm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.impl.LinearKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.model.SVMModel;
import com.ml.hw7.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class SMOImpl {

	public static SVMKernel kernel;
	private Map<Integer, Double> fxCache;
	private double alphas[];
	private double b;
	
	public SMOImpl(int dataSize) {
		fxCache = new HashMap<Integer, Double>();
		alphas = new double[dataSize];
		SMOImpl.kernel = new LinearKernel(dataSize);
	}
	
	
	public SVMModel train(DataSet trainingData, Map<String, Object> additionalData) throws Exception {
		List<Data> datas = trainingData.getData();
		
		double c = (double) additionalData.get(SVMUtil.SVM_PARAMETER_C);
		double tolerance = (double) additionalData.get(SVMUtil.SVM_PARAMETER_TOLERANCE);
		int maxPasses = (int) additionalData.get(SVMUtil.SVM_PARAMETER_PASSES);
		
		int pass=0;
		while(pass < maxPasses) {
			System.out.println("pass"+ pass);
			int numberOfChangedAlphas = 0;
			for(int indexX1=0; indexX1 < trainingData.dataSize(); indexX1++) {
				numberOfChangedAlphas = runAlphaOptimization(c,tolerance, indexX1, datas, numberOfChangedAlphas, additionalData);
			}
			System.out.println("changed alphas: "+numberOfChangedAlphas);
			if(numberOfChangedAlphas == 0) {
				pass++;
			} else {
				pass = 0;
			}
		}
		
		SVMModel model = new SVMModel();
		model.setAlphas(alphas);
		model.setB(b);
		System.out.println("");
		return model;
	}

	private int runAlphaOptimization(double c, double tolerance, int indexX1,
			List<Data> datas, int numberOfChangedAlphas,  Map<String, Object> additionalData) throws Exception {
		//System.out.print("*");
		double errorX1 = calculateError(alphas, b, indexX1, datas, additionalData);
		double labelX1 = datas.get(indexX1).labelValue();
		boolean performOptimization = checkOptimizationCOnditions(labelX1, errorX1, tolerance, alphas[indexX1], c);
		if(performOptimization) {
			int indexX2 = selectRandomJ(indexX1, datas.size());
			double labelX2 = datas.get(indexX2).labelValue();
			double errorX2 = calculateError(alphas, b, indexX2, datas, additionalData);
			double oldAlphaX1 = alphas[indexX1];
			double oldAlphaX2 = alphas[indexX2];
			double oldB = b;
			double low = calculateL(labelX1, labelX2, oldAlphaX1, oldAlphaX2, c);
			double high = calculateH(labelX1, labelX2, oldAlphaX1, oldAlphaX2, c);
			if(low == high) {
				return numberOfChangedAlphas;
			}
			double eta = calculateETA(datas, indexX1, indexX2, additionalData);
			if(eta >= 0) {
				return numberOfChangedAlphas;
			}
			double alphaJ = calculateAlphaJ(alphas[indexX2], labelX2, errorX1, errorX2, eta, low, high);
			alphas[indexX2] = alphaJ;
			
			if(Math.abs(alphas[indexX2] - oldAlphaX2) < .00001) {
				refreshFXCache(datas, indexX1, indexX2, oldAlphaX1, oldAlphaX2, oldB, additionalData);
				return numberOfChangedAlphas;
			}
			alphas[indexX1] = calculateAlphaI(alphas[indexX1], labelX1, labelX2, oldAlphaX2, alphas[indexX2]);
			double b1 = calculateB1(errorX1, labelX1, labelX2, oldAlphaX1, oldAlphaX2, datas, indexX1, indexX2, additionalData);
			double b2 = calculateB2(errorX2, labelX1, labelX2, oldAlphaX1, oldAlphaX2, datas, indexX1, indexX2, additionalData);
			b = calculateB(b1, b2, alphas[indexX1], alphas[indexX2], c);
			refreshFXCache(datas, indexX1, indexX2, oldAlphaX1, oldAlphaX2, oldB, additionalData);
			numberOfChangedAlphas++;
			//System.out.print(".");
		}
		return numberOfChangedAlphas;
	}
	
	private void refreshFXCache(List<Data> datas, int indexX1, int indexX2, double oldAlphaX1, double oldAlphaX2, 
			double oldB, Map<String, Object> additionalData) throws Exception {
		boolean cacheEnabled = (boolean) additionalData.get(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED);
		if(cacheEnabled) {
			for(int i=0; i < datas.size(); i++) {
				Data dataI = datas.get(i);
				Data dataA = datas.get(indexX1);
				Data dataB = datas.get(indexX2);
				Double cachedValue = fxCache.get(i);
				if(cachedValue != null) {
					double value = cachedValue;
					value+= (alphas[indexX1] - oldAlphaX1) * dataA.labelValue() * kernel.evaluateKernel(dataI, dataA, i, indexX1, additionalData);
					value+= (alphas[indexX2] - oldAlphaX2) * dataB.labelValue() * kernel.evaluateKernel(dataI, dataB, i, indexX2, additionalData);
					value+= (b - oldB);
					fxCache.put(i, value);
				}
			}
		}
	}

	private double calculateB(double b1, double b2, double alphaX1, double alphaX2, double c) {
		if(alphaX1 > 0 && alphaX1 < c) {
			return b1;
		}
		if(alphaX2 > 0 && alphaX2 < c) {
			return b2;
		}
		return (b1+b2)/2;
	}

	private double calculateB2(double errorX2, double labelX1, double labelX2, double oldAlphaX1, double oldAlphaX2, List<Data> datas,
			int indexX1, int indexX2, Map<String, Object> additionalData) throws Exception {
		Data x1 = datas.get(indexX1);
		Data x2 = datas.get(indexX2);
		double alphaX1 = alphas[indexX1];
		double alphaX2 = alphas[indexX2];
		
		double b2 = b - errorX2;
		b2-= labelX1 * (alphaX1 - oldAlphaX1) * kernel.evaluateKernel(x1, x2, indexX1, indexX2, additionalData);
		b2-= labelX2 * (alphaX2 - oldAlphaX2) * kernel.evaluateKernel(x2, x2, indexX2, indexX2, additionalData);
		return b2;
	}
	
	
	private double calculateB1(double errorX1, double labelX1, double labelX2, double oldAlphaX1, double oldAlphaX2, List<Data> datas, 
			int indexX1, int indexX2, Map<String, Object> additionalData) throws Exception {
		Data x1 = datas.get(indexX1);
		Data x2 = datas.get(indexX2);
		double alphaX1 = alphas[indexX1];
		double alphaX2 = alphas[indexX2];
		
		double b1 = b - errorX1;
		b1-= labelX1 * (alphaX1 - oldAlphaX1) * kernel.evaluateKernel(x1, x1, indexX1, indexX1, additionalData);
		b1-= labelX2 * (alphaX2 - oldAlphaX2) * kernel.evaluateKernel(x1, x2, indexX1, indexX2, additionalData);
		return b1;
	}

	private double calculateAlphaI(double alphaX1, double labelX1, double labelX2, double oldAlphaX2, double alphaX2) {
		double alphaI = alphaX1 + (labelX1 * labelX2) * (oldAlphaX2 - alphaX2);
		return alphaI;
	}

	private double calculateAlphaJ(double alphaX2, double labelX2, double errorX1, double errorX2, double eta,
			double low, double high) {
		double alphaJ = alphaX2 - (labelX2) * (errorX1 - errorX2)/eta;
		alphaJ = clipAlphaJ(alphaJ, high, low);
		return alphaJ;
	}

	private double clipAlphaJ(double alphaJ, double high, double low) {
		double result = alphaJ;
		if(alphaJ > high) {
			result = high;
		}
		if(alphaJ < low) {
			result = low;
		}
		return result;
	}

	private double calculateETA(List<Data> datas, int indexX1, int indexX2, Map<String, Object> additionalData) throws Exception {
		Data dataX1 = datas.get(indexX1);
		Data dataX2 = datas.get(indexX2);
		double eta = 0;
		eta+= 2 * kernel.evaluateKernel(dataX1, dataX2, indexX1, indexX2, additionalData);
		eta-= kernel.evaluateKernel(dataX1, dataX1, indexX1, indexX1, additionalData);
		eta-= kernel.evaluateKernel(dataX2, dataX2, indexX2, indexX2, additionalData);
		return eta;
	}

	private double calculateH(double labelX1, double labelX2, double alphaX1, double alphaX2, double c) {
		if(labelX1 == labelX2) {
			return Math.min(c, alphaX1+alphaX2);
		}
		return Math.min(c, c + alphaX2 - alphaX1);
	}

	private double calculateL(double labelX1, double labelX2, double alphaX1, double alphaX2, double c) {
		if(labelX1 == labelX2) {
			return Math.max(0, alphaX1 + alphaX2 - c);
		}
		return Math.max(0, alphaX2 - alphaX1); 
	}

	private int selectRandomJ(int indexX1, int range) {
		Random random = new Random();
		int indexX2 = 0;
		do {
			indexX2 = random.nextInt(range);
		} while (indexX1 == indexX2);
		return indexX2;
	}

	private boolean checkOptimizationCOnditions(double label, double error, double tolerance, double alpha, double c) {
		double yMultError = label * error;
		return ((yMultError < -tolerance) && (alpha < c)) || ((yMultError > tolerance) && alpha > 0);
	}

	private double calculateFx(double[] alphas, double b, int indexX1, List<Data> datas,  Map<String, Object> additionalData) throws Exception {
		Double value = null;
		boolean cacheEnabled = (boolean) additionalData.get(SVMUtil.SSVM_PARAMETERS_FX_CACHE_ENABLED);
		if(cacheEnabled) {
			value = fxCache.get(indexX1);
			if(value == null) {
				value = computeFX(datas, alphas ,indexX1, b, additionalData);
				fxCache.put(indexX1, value);
			}
		} else {
			value = computeFX(datas, alphas, indexX1, b, additionalData);
		}
		return value;
	}
	

	
	private Double computeFX(List<Data> datas, double[] alphas, int indexX1, double b, Map<String, Object> additionalData) throws Exception {
		double value = 0;
		Data data1 = datas.get(indexX1);
		for(int indexX2 = 0; indexX2 < datas.size(); indexX2++) {
			Data data2 = datas.get(indexX2);
			value+= alphas[indexX2] * data2.labelValue() * kernel.evaluateKernel(data1, data2, indexX1, indexX2, additionalData);
		}
		value+= b;
		return value;
	}

	private double calculateError(double[] alphas, double b, int indexX1, List<Data> datas,  Map<String, Object> additionalData) 
			throws Exception {
		Data data = datas.get(indexX1);
		double error = calculateFx(alphas, b, indexX1, datas, additionalData) - data.labelValue();
		return error;
	}
}