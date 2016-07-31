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
public class CosineDistanceKernel extends AbstractKernel {

	public CosineDistanceKernel(int size) {
		super(size, SVMUtil.COSINE_SIMILARITY);
	}
	
	@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		return 1 - super.evaluateKernel(x1, x2, indexX1, indexX2, additionalData);
	}
}
