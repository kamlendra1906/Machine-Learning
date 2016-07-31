/**
 * 
 */
package com.ml.hw6.classifier.bagging;

/**
 * @author kkumar
 *
 */
public interface TreeBuilder {
	
	public TreeNode buildTree(TreeNode node) throws Exception;

}
