/**
 * 
 */
package src.com.ml.hw3.classifier.stats;

import Jama.Matrix;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class MixtureModel {

	private int modelNumber;
	private int dataSize;
	private Matrix z;
	private GaussianModel[] models;
	private double[] modelProbability;
	
	public MixtureModel(int modelNumber, int dataSize) {
		this.modelNumber = modelNumber;
		this.dataSize = dataSize;
		models = new GaussianModel[this.modelNumber];
		modelProbability = new double[this.modelNumber];
		z = new Matrix(this.dataSize, this.modelNumber);
		initializeModelProbability();
		initializeZ();
	}

	/**
	 * @return the modelNumber
	 */
	public int getModelNumber() {
		return modelNumber;
	}

	/**
	 * @param modelNumber the modelNumber to set
	 */
	public void setModelNumber(int modelNumber) {
		this.modelNumber = modelNumber;
	}

	/**
	 * @return the dataSize
	 */
	public int getDataSize() {
		return dataSize;
	}

	/**
	 * @param dataSize the dataSize to set
	 */
	public void setDataSize(int dataSize) {
		this.dataSize = dataSize;
	}

	/**
	 * @return the z
	 */
	public Matrix getZ() {
		return z;
	}

	/**
	 * @param z the z to set
	 */
	public void setZ(Matrix z) {
		this.z = z;
	}

	/**
	 * @return the models
	 */
	public GaussianModel[] getModels() {
		return models;
	}

	/**
	 * @param models the models to set
	 */
	public void setModels(GaussianModel[] models) {
		this.models = models;
	}

	/**
	 * @return the modelProbability
	 */
	public double[] getModelProbability() {
		return modelProbability;
	}

	/**
	 * @param modelProbability the modelProbability to set
	 */
	public void setModelProbability(double[] modelProbability) {
		this.modelProbability = modelProbability;
	}	
	
	private void initializeModelProbability() {
		double total = 0;
		for(int model=0; model < this.modelNumber; model++) {
			modelProbability[model] = Math.random(); 
			total+= modelProbability[model];
		}
		ClassifierUtil.normalizeProbability(modelProbability, total);
	}

	private void initializeZ() {
		for(int dataRow = 0; dataRow < this.dataSize; dataRow++) {
			int index = (int) ((Math.random() * 131) % this.modelNumber);
			this.z.set(dataRow, index, 1);
		}
	}
}