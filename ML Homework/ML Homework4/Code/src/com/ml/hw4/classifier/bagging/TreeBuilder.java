/**
 * 
 */
package com.ml.hw4.classifier.bagging;

/**
 * @author kkumar
 *
 */
public interface TreeBuilder {
	
	public TreeNode buildTree(TreeNode node) throws Exception;

}
