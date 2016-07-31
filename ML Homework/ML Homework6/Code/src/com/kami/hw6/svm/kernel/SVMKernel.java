package com.kami.hw6.svm.kernel;

import java.util.Map;

import com.ml.hw6.data.Data;

public interface SVMKernel {

	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData) throws Exception;
}
