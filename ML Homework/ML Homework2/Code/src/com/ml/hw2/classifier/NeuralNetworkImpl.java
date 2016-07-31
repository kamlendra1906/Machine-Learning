/**
 * 
 */
package com.ml.hw2.classifier;

import java.util.List;

import com.ml.hw2.data.Node;
import com.ml.hw2.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class NeuralNetworkImpl {

	private Node[] inputLayer;
	private Node[] hiddenLayer;
	private Node[] outputLayer;
	
	private double[][] wInputToHidden;
	private double[][] wHiddenToOutput;
	
	private int inputNodeNum;
	private int hiddenNodeNum;
	private int outputNodeNum;
	
	public NeuralNetworkImpl(int inputNodeNum, int hiddenNodeNum, int outputNodeNum) {
		this.inputNodeNum = inputNodeNum;
		this.hiddenNodeNum = hiddenNodeNum;
		this.outputNodeNum = outputNodeNum;
		inputLayer = new Node[inputNodeNum];
		hiddenLayer = new Node[hiddenNodeNum];
		outputLayer = new Node[outputNodeNum];
		wInputToHidden = new double[inputNodeNum][hiddenNodeNum];
		wHiddenToOutput = new double[hiddenNodeNum][outputNodeNum];
		initializeNetworkWeights();
		initializeHiddenAndOutputLayerNodes();
	}

	private void initializeNetworkWeights() {
		// initialize input to hidden layer weight
		for(int row=0; row < inputNodeNum; row++) {
			for(int col=0; col < hiddenNodeNum; col++) {
				if(col == 0) {
					wInputToHidden[row][col] = 0;
				} else {
					wInputToHidden[row][col] = Math.random();
				}
			}
		}
		
		// initialize hidden to output layer weight
		for(int row=0; row < hiddenNodeNum; row++) {
			for(int col=0; col < outputNodeNum; col++) {
				wHiddenToOutput[row][col] = Math.random();
			}
		}
	}
	
	private void initializeHiddenAndOutputLayerNodes() {
		// initialize hidden nodes
		for(int count=0; count < hiddenNodeNum; count++) {
			if(count == 0) {
				hiddenLayer[count] = new Node(1,1,0);
			} else {
				hiddenLayer[count] = new Node();
			}
		}
		
		// initialize input nodes
		for(int count=0; count < inputNodeNum; count++) {
			if(count == 0) {
				inputLayer[count] = new Node(1, 1, 0);
			} else {
				inputLayer[count] = new Node();
			}
		}
		
		// initialize output nodes
		for(int count=0; count < outputNodeNum; count++) {
			outputLayer[count] = new Node();
		}
	}
	
	public void train(List<double[]> trainingDataSet, double threshold, double lambda) {
		double oldError = calculateNetworkMSE(trainingDataSet);
		if(trainingDataSet != null && trainingDataSet.size() > 0) {
			int iterations = 0;
			while(true) {
				iterations++;
				for(double[] trainingDataPoint : trainingDataSet) {
					doNeuralNetworkLearning(trainingDataPoint, lambda);
					//System.out.print(ClassifierUtil.printArray(trainingDataPoint));
					//System.out.println(" "+printLayerOutput(hiddenLayer, true)+"        "+printLayerOutput(outputLayer, false));

				}
				//System.out.println("\n\n");
				double error = calculateNetworkMSE(trainingDataSet);
				/**System.out.println("iteration="+ iterations+ "  MSE New= "+error+  "  MSE Old="+oldError);
				if(Math.abs(oldError - error) < threshold) {
					System.out.println(iterations);
					break;
				}*/
				oldError = error;
				if(iterations > 500000) {
					break;
				}
			}
		}
	}

	private void doNeuralNetworkLearning(double[] trainingDataPoint , double lambda) {
		importTrainingDataIntoNetwork(trainingDataPoint);
		updateHiddenNetworkNetInputAndOutPut();
		updateOutputLayerNetInputAndOutput();
		backPropagateError(lambda);
		
	}
	
	private void importTrainingDataIntoNetwork(double[] trainingDataPoint) {
		for(int count=0; count < trainingDataPoint.length; count++) {
			// set the input data
			inputLayer[count+1].setNetInput(trainingDataPoint[count]);
			inputLayer[count+1].setOutput(trainingDataPoint[count]);
			
			// set the output label
			outputLayer[count].setActualOutput(trainingDataPoint[count]);
		}
	}
	
	private void updateHiddenNetworkNetInputAndOutPut() {
		for(int hiddenRow = 1; hiddenRow < hiddenNodeNum; hiddenRow++) {
			hiddenLayer[hiddenRow].setNetInput(calculateNetInputForHiddenLayerNode(hiddenRow));
		}
	}

	private double calculateNetInputForHiddenLayerNode(int hiddenRow) {
		double netInput = 0;
		for(int inputRow = 0; inputRow < inputNodeNum; inputRow++) {
			netInput+= inputLayer[inputRow].getOutput() * wInputToHidden[inputRow][hiddenRow];
		}
		return netInput;
	}
	
	private void updateOutputLayerNetInputAndOutput() {
		for(int outputRow = 0; outputRow < outputNodeNum; outputRow++) {
			outputLayer[outputRow].setNetInput(calculateNetInputForOutputLayerNode(outputRow));
		}
	}

	private double calculateNetInputForOutputLayerNode(int outputRow) {
		double netInput = 0;
		for(int hiddenRow = 0; hiddenRow < hiddenNodeNum; hiddenRow++) {
			netInput+= hiddenLayer[hiddenRow].getOutput() * wHiddenToOutput[hiddenRow][outputRow];
		}
		return netInput;
	}

	private void backPropagateError(double lambda) {
		double[] errorOutputLayer = calculateOutputLayerError();
		double[] errorHiddenLayer = calculateHiddenLayerError(errorOutputLayer);
		updateHiddenToOutputLayerWeights(errorOutputLayer, errorHiddenLayer, lambda);
		updateInputToHiddenLayerWeights(errorOutputLayer, errorHiddenLayer, lambda);
	}

	private double[] calculateOutputLayerError() {
		double[] errorOutputLayer = new double[outputNodeNum];
		for(int outputRow=0; outputRow < outputNodeNum; outputRow++) {
			Node node = outputLayer[outputRow];
			errorOutputLayer[outputRow] = node.getGradiant() * node.getError();
		}
		return errorOutputLayer;
	}
	
	private double[] calculateHiddenLayerError(final double[] errorOutputLayer) {
		double[] errorHiddenLayer = new double[hiddenNodeNum];
		
		for(int hiddenRow = 0; hiddenRow < hiddenNodeNum; hiddenRow++) {
			double error = 0;
			Node node = hiddenLayer[hiddenRow];
			for(int outputRow = 0; outputRow < outputNodeNum; outputRow++) {
				error+= errorOutputLayer[outputRow] * wHiddenToOutput[hiddenRow][outputRow];
			}
			error*= node.getGradiant();
			errorHiddenLayer[hiddenRow] = error;
		}
		return errorHiddenLayer;
	}
	
	private void updateHiddenToOutputLayerWeights(double[] errorOutputLayer, double[] errorHiddenLayer, double lambda) {
		for(int hiddenRow = 0; hiddenRow < hiddenNodeNum; hiddenRow++) {
			Node node = hiddenLayer[hiddenRow];
			for(int outputRow = 0; outputRow < outputNodeNum; outputRow++) {
				wHiddenToOutput[hiddenRow][outputRow]+= errorOutputLayer[outputRow] * node.getOutput() * lambda;
			}
		}
	}

	private void updateInputToHiddenLayerWeights(double[] errorOutputLayer, double[] errorHiddenLayer, double lambda) {
		for(int inputRow=0; inputRow < inputNodeNum; inputRow++) {
			Node node = inputLayer[inputRow];
			for(int hiddenRow=0; hiddenRow < hiddenNodeNum; hiddenRow++) {
				if(hiddenRow == 0) {
					continue;
				}
				wInputToHidden[inputRow][hiddenRow]+= errorHiddenLayer[hiddenRow] * node.getOutput() * lambda;
			}
		}
	}

	private double calculateNetworkMSE(List<double[]> trainingDataSet) {
		double error = 0;
		for(double[] trainingDataPoint : trainingDataSet) {
			error+= calculateMSEForTestPoint(trainingDataPoint);
		}
		return error/trainingDataSet.size();
	}

	private double calculateMSEForTestPoint(double[] trainingDataPoint) {
		importTrainingDataIntoNetwork(trainingDataPoint);
		updateHiddenNetworkNetInputAndOutPut();
		updateOutputLayerNetInputAndOutput();
		double error = 0;
		for(int outputRow = 0; outputRow < outputNodeNum; outputRow++) {
			Node node = outputLayer[outputRow];
			error+= Math.pow(node.getError(),2);
		}
		return error/2;
	}
	
	private String printLayerOutput(Node[] layer, boolean hiddenLayer) {
		StringBuilder builder = new StringBuilder();
		for(int hiddenRow=0; hiddenRow < layer.length; hiddenRow++) {
			if(hiddenLayer && hiddenRow == 0) {
				continue;
			}
			builder.append(layer[hiddenRow].getOutput() > 0.5 ? 1 : 0+",");
		}
		return builder.toString();
	}

	public void test(List<double[]> testDataSet) {
		System.out.println("Testing Begins\n\n");
		for(double[] testDataPoint : testDataSet) {
			importTrainingDataIntoNetwork(testDataPoint);
			updateHiddenNetworkNetInputAndOutPut();
			updateOutputLayerNetInputAndOutput();
			System.out.print(ClassifierUtil.printArray(testDataPoint));
			System.out.print(" "+printLayerOutput(hiddenLayer, true)+"        "+printLayerOutput(outputLayer, false)+"\n");
		}
	}
}