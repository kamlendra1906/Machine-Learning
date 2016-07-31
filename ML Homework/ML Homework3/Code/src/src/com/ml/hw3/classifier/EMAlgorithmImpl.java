/**
 * 
 */
package src.com.ml.hw3.classifier;

import java.util.List;

import Jama.Matrix;
import src.com.ml.hw3.classifier.stats.GaussianModel;
import src.com.ml.hw3.classifier.stats.MixtureModel;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class EMAlgorithmImpl {

	private MixtureModel mixtureModel;
	private List<Matrix> trainingData;
	
	public EMAlgorithmImpl(MixtureModel model, List<Matrix> trainingData) {
		this.mixtureModel = model;
		this.trainingData = trainingData;
	}
	
	public void runEMAlgorithm() {
		int iteration = 0;
		while(true) {
			iteration++;
			runEStep();
			runMStep();
			if(iteration == 1000) {
				break;
			}
		}
	}

	private void runEStep() {
		int modelNumber = mixtureModel.getModelNumber();
		int dataSize = trainingData.size();
		GaussianModel[] models = mixtureModel.getModels();
		double[] modelProbability = mixtureModel.getModelProbability();
		Matrix z = mixtureModel.getZ();
		
		for(int dataRow = 0; dataRow < dataSize; dataRow++) {
			
			Matrix x = trainingData.get(dataRow);
			double[] dataModelProbabilityArray = new double[modelNumber];
			double totalProbability = 0;
		
			for(int model=0; model < modelNumber; model++) {
				double dataModelProbability = models[model].getGenerativeProbabilityOfPoint(x) * modelProbability[model];
				dataModelProbabilityArray[model] = dataModelProbability;
				totalProbability+= dataModelProbability;
			}
			
			ClassifierUtil.normalizeProbability(dataModelProbabilityArray, totalProbability);
			
			for(int model=0; model < modelNumber; model++) {
				z.set(dataRow, model, dataModelProbabilityArray[model]);
			}
		}
	}
	
	private void runMStep() {
		int modelNumber = mixtureModel.getModelNumber();
		GaussianModel[] models = mixtureModel.getModels();
		Matrix z = mixtureModel.getZ();
		double[] modelProbability = mixtureModel.getModelProbability();
		
		for(int model= 0; model < modelNumber; model++) {
			Matrix covarriance = models[model].getCovariance();
			Matrix mean = models[model].getMean();
			
			Matrix tempCovarriance = new Matrix(models[0].getCovariance().getRowDimension(), models[0].getCovariance().getRowDimension());
			Matrix tempMean = new Matrix(models[0].getMean().getRowDimension(), models[0].getMean().getColumnDimension());
			
			double sumZim = 0;
			
			for(int dataRow = 0; dataRow < this.trainingData.size(); dataRow++) {
				Matrix x = trainingData.get(dataRow);
				double Zim = z.get(dataRow, model);
				sumZim += Zim;
				
				Matrix iterationCovariance  = x.minus(mean).times(x.minus(mean).transpose()).times(Zim);
				Matrix iterationMean =  x.times(Zim);
				tempCovarriance = tempCovarriance.plus(iterationCovariance);
				tempMean = tempMean.plus(iterationMean);
			}
			tempCovarriance = tempCovarriance.times(1/sumZim);
			tempMean = tempMean.times(1/sumZim);
			models[model].setCovariance(tempCovarriance);
			models[model].setMean(tempMean);
			modelProbability[model] = sumZim / this.trainingData.size();
		}
	}
}