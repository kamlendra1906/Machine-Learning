/**
 * 
 */
package com.kami.hw6.svm.kernel.impl;

import java.util.Map;

import com.ml.hw6.data.Data;

/**
 * @author kkumar
 *
 */
public class LinearKernel extends AbstractKernel {

	public LinearKernel(int size) {
		super(size);
	}
	
	@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		return super.evaluateKernel(x1, x2, indexX1, indexX2, additionalData);
	}
}