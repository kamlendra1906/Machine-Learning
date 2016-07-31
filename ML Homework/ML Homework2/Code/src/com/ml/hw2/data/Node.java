/**
 * 
 */
package com.ml.hw2.data;

/**
 * @author kkumar
 *
 */
public class Node {

	private double netInput;
	private double output;
	private double actualOutput;
	
	public Node() {
		this.netInput = 0;
		this.output = 0;
		this.actualOutput = 0;
	}
	
	public Node(double netInput, double output, double actualOutput) {
		this.netInput = netInput;
		this.output = output;
		this.actualOutput = actualOutput;
	}

	/**
	 * @return the netInput
	 */
	public double getNetInput() {
		return netInput;
	}

	/**
	 * @param netInput the netInput to set
	 */
	public void setNetInput(double netInput) {
		this.netInput = netInput;
		this.calculateOutput();
	}

	/**
	 * @return the output
	 */
	public double getOutput() {
		return output;
	}

	/**
	 * @param output the output to set
	 */
	public void setOutput(double output) {
		this.output = output;
	}

	public void calculateOutput() {
		this.setOutput(getLogisticRegressionOutput());
	}
	
	private double getLogisticRegressionOutput() {
		return 1/(1+Math.exp(-netInput));
	}

	/**
	 * @return the actualOutput
	 */
	public double getActualOutput() {
		return actualOutput;
	}

	/**
	 * @param actualOutput the actualOutput to set
	 */
	public void setActualOutput(double actualOutput) {
		this.actualOutput = actualOutput;
	}
	
	public double getGradiant() {
		return output * (1-output);
	}
	
	public double getError() {
		return actualOutput - output;
	}
	
	@Override
	public String toString() {
		return "netInput= "+ netInput+"  output="+output+"  actual_output="+actualOutput;
	}
}