/**
 * 
 */
package com.kami.hw7.svm.kernel.impl;

import java.util.Map;

import com.ml.hw7.data.Data;
import com.ml.hw7.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class PolynomialKernel extends AbstractKernel {

	public PolynomialKernel(int size) {
		super(size, SVMUtil.DOT_PRODUCT_SIMILARITY);
	}

	@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		double a = (double) additionalData.get(SVMUtil.SVM_PARAMETER_ALPHA);
		double b = (double) additionalData.get(SVMUtil.SVM_PARAMETER_BETA);
		double d = (double) additionalData.get(SVMUtil.SVM_PARAMETER_DEGREE);
		
		double dotProduct = super.evaluateKernel(x1, x2, indexX1, indexX2, additionalData);
		
		double result = a*dotProduct;
		result+= b;
		return Math.pow(result, d);
	}
}
