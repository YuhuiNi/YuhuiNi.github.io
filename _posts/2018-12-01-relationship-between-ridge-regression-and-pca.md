---
layout: post
title: Relationship between Ridge regression and PCA
---

I believe everyone who is familiar with machine learning should know about ridge regression(or \\(L_2\\) norm regularized least square, or Linear regression with Gaussian prior) and PCA. Actually, these are concepts we learnt at the beginning of our study in machine learning.

Why do we want to study the relationship between ridge regression and PCA? This problem arises from linear regression.

### 1.Linear regression

$$Y=X\beta +e$$

$$Y\in \mathbb{R^{n\times 1}}, X\in \mathbb{R^{n\times p}}, \beta \in \mathbb{R^{p \times 1}}$$

If X has full column rank, then \\(\beta\\) has a close form solution
$$\beta=argmin_{\beta}||Y-X\beta||_2^2=(X^TX)^{-1}X^T Y$$


However, if \\(rank(X)<p\\), we cannot get an analytical solution of \\(\beta\\). In general, there are two ways to solve this problem. 

* we can implement PCA on \\(X\\) and then do linear regression.
* we can add regularization term(\\(L1,L2\\)) on loss function.


### 2.PCA

\\(X_{PCA}\\) can be obtained from SVD, specially

$$X=U\Sigma V^T$$

where \\(\Sigma\\) is a diagnoal matrix.

$$\Sigma=
\left[\begin{array}{ccc} 
    \sigma_1 &        &  \\ 
     &   \ddots   & \\ 
     &   & \sigma_p
\end{array}\right]
$$

where \\(\sigma_i\\) is the singular value and \\(\forall i<j, \sigma_i\geq \sigma_j\\)

If we want to keep \\(k\\) largest principal components of X, then 
$$\begin{align}
X_{PCA}&=
\end{align}U_k\Sigma_k=U_k
\left[\begin{array}{ccc} 
    \sigma_1 &        &  \\ 
     &   \ddots   & \\ 
     &   & \sigma_k
\end{array}\right] 
$$

If we plug \\(X_{PCA}\\) into \\(\hat{Y}=X\hat{\beta}=X(X^TX)^{-1}X^TY\\), then we have
$$
\hat{Y}_{PCA}=U_kU_k^TY= U \left[\begin{array}{cccccc}
    1 & &  & & &\\ 
     &   \ddots   & & & &\\ 
     &   & 1 & & &\\
     & & & 0& &\\
     & & & &\ddots &\\
     & & & & &0\\
 \end{array}\right] U^TY=\sum_{j=1}^{k}u_ju_j^TY
$$

### 3.Ridge regression
For ridge regression, our solution is

$$\beta_{ridge}=argmin_{\beta}||Y-X\beta||_2^2+\lambda||\beta||_2^2=(X^TX+\lambda I)^{-1}X^TY$$

According to
$$Y_{ridge}=X\beta_{ridge}$$

plug \\(\beta_{ridge}\\) and \\(X=U\Sigma V^T\\) into above equation, we have

$$\begin{align}
\hat{Y}_{ridge} &= X(X^TX+\lambda I)^{-1}X^TY \\
               &= U\Sigma V^T(V\Sigma^2V^T+\lambda VIV^T)^{-1}V\Sigma U^TY \\
               &= U\Sigma V^TV(\Sigma^2+\lambda I)^{-1}V^TV \Sigma U^TY \\
               &= U\Sigma(\Sigma^2+\lambda I)^{-1}\Sigma U^TY \\
               &= \sum_{j=1}^{p}u_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}u_j^TY
\end{align}$$

### 4.Comparison

$$\hat{Y}_{PCA}=(\sum_{j=1}^{k}u_j\cdot 1 \cdot u_j^T+\sum_{j=k+1}^{p}u_j\cdot 0 \cdot u_j^T)Y$$

$$\hat{Y}_{ridge}=\sum_{j=1}^{p}u_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}u_j^TY$$

When we put these two equaitons together, we can see their relationship clearly.

* for ridge regression, regularization term \\(\lambda \\) has different impact on different singular values. When \\(\sigma_j\\) is small, \\(\frac{\sigma_j^2}{\sigma_j^2+\lambda}\\) is also very small. On the contrary, \\(\frac{\sigma_j^2}{\sigma_j^2+\lambda}\\) tends to be 1 when \\(\sigma_j\\) is large. 

* for PCA, it sets all dimensions with small singular values to be 0 and remaining other dimensions to be 1. 

Therefore, ridge regression is a soft PCA regression in fact. They both intend to solve the multi-collinearity in order to improve the model fittness.

### Reference

1.[The Elements of 
Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)