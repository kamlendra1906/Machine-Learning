/**
 * 
 */
package src.com.ml.hw3.classifier.stats;

import org.ejml.alg.dense.linsol.svd.SolvePseudoInverseSvd;
import org.ejml.data.DenseMatrix64F;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class GDAModel {
	
	private double probabilityOfOne;
	private Matrix mueZero;
	private Matrix mueOne;
	private Matrix coVarriance;
	private Matrix coVarrianceClassZero;
	private Matrix coVarrianceClassOne;
	private Matrix invCoVarriance;
	private Matrix invCoVarrianceClassZero;
	private Matrix invCoVarrianceClassOne;
	
	public GDAModel() {
	}

	/**
	 * @return the probabilityOfOne
	 */
	public double getProbabilityOfOne() {
		return probabilityOfOne;
	}

	/**
	 * @param probabilityOfOne the probabilityOfOne to set
	 */
	public void setProbabilityOfOne(double probabilityOfOne) {
		this.probabilityOfOne = probabilityOfOne;
	}

	/**
	 * @return the mueZero
	 */
	public Matrix getMueZero() {
		return mueZero;
	}

	/**
	 * @param mueZero the mueZero to set
	 */
	public void setMueZero(Matrix mueZero) {
		this.mueZero = mueZero;
	}

	/**
	 * @return the mueOne
	 */
	public Matrix getMueOne() {
		return mueOne;
	}

	/**
	 * @param mueOne the mueOne to set
	 */
	public void setMueOne(Matrix mueOne) {
		this.mueOne = mueOne;
	}

	/**
	 * @return the coVarriance
	 */
	public Matrix getCoVarriance() {
		return coVarriance;
	}

	/**
	 * @param coVarriance the coVarriance to set
	 */
	public void setCoVarriance(Matrix coVarriance) {
		this.coVarriance = coVarriance;
		this.setInvCoVarriance(this.getInverseMatrix(this.coVarriance));
	}

	/**
	 * @return the coVarrianceClassZero
	 */
	public Matrix getCoVarrianceClassZero() {
		return coVarrianceClassZero;
	}

	/**
	 * @param coVarrianceClassZero the coVarrianceClassZero to set
	 */
	public void setCoVarrianceClassZero(Matrix coVarrianceClassZero) {
		this.coVarrianceClassZero = coVarrianceClassZero;
		this.setInvCoVarrianceClassZero(this.getInverseMatrix(this.coVarrianceClassZero));
	}

	/**
	 * @return the coVarrianceClassOne
	 */
	public Matrix getCoVarrianceClassOne() {
		return coVarrianceClassOne;
	}

	/**
	 * @param coVarrianceClassOne the coVarrianceClassOne to set
	 */
	public void setCoVarrianceClassOne(Matrix coVarrianceClassOne) {
		this.coVarrianceClassOne = coVarrianceClassOne;
		this.setInvCoVarrianceClassOne(getInverseMatrix(this.coVarrianceClassOne));
	}

	/**
	 * @return the invCoVarriance
	 */
	public Matrix getInvCoVarriance() {
		return invCoVarriance;
	}

	/**
	 * @param invCoVarriance the invCoVarriance to set
	 */
	public void setInvCoVarriance(Matrix invCoVarriance) {
		this.invCoVarriance = invCoVarriance;
	}

	/**
	 * @return the invCoVarrianceClassZero
	 */
	public Matrix getInvCoVarrianceClassZero() {
		return invCoVarrianceClassZero;
	}

	/**
	 * @param invCoVarrianceClassZero the invCoVarrianceClassZero to set
	 */
	public void setInvCoVarrianceClassZero(Matrix invCoVarrianceClassZero) {
		this.invCoVarrianceClassZero = invCoVarrianceClassZero;
	}

	/**
	 * @return the invCoVarrianceClassOne
	 */
	public Matrix getInvCoVarrianceClassOne() {
		return invCoVarrianceClassOne;
	}

	/**
	 * @param invCoVarrianceClassOne the invCoVarrianceClassOne to set
	 */
	public void setInvCoVarrianceClassOne(Matrix invCoVarrianceClassOne) {
		this.invCoVarrianceClassOne = invCoVarrianceClassOne;
	}

	private Matrix getInverseMatrix(Matrix matrixToBeInversed) {
		DenseMatrix64F matrix = new DenseMatrix64F(matrixToBeInversed.getArray());
		DenseMatrix64F inverseMatrix = new DenseMatrix64F(matrix.getNumRows(), matrix.getNumCols());
		
		SolvePseudoInverseSvd svd = new SolvePseudoInverseSvd();
		svd.setA(matrix);
		svd.invert(inverseMatrix);
		
		double[][] inversedMatrixData = new double[matrix.getNumRows()][matrix.getNumRows()];
		for(int row=0; row < matrix.getNumRows(); row++) {
			for(int col=0; col < matrix.getNumCols(); col++) {
				inversedMatrixData[row][col] = inverseMatrix.get(row, col);
			}
		}
		return new Matrix(inversedMatrixData);
	}

}
