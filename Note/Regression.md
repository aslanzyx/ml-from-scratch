# Regressions

## Linear Regression
- Regrssions are to find out eelations between (numberic) variables
- Linear regression makes predictions $\hat{y_i}$ using a linear function of $x_i$
- $\hat{y_i}=w x_i$
- $w$ is the weight/regression coefficient of $x_i$

### Terminology
- Data points = examples
- Inputs = features
- Output = outcomes
- Regression coefficient = weight

### Objective Least Square
- **Model**: described by a vector of weights
- **Prodections**: by $\hat{y_i}=w^Tx_i$
- **Sum of Squre Error:** $f(w)=\sum_i{(\hat{y_i}-y_i)^2}$

### Optimization (with no bias)
- To minimize SSE we want $\nabla f=\vec{0}$ and $\nabla^2 f > \vec{0}$
- Digression: Optimizing $f(x)$ is the same as optimizing $a\cdot f(x)+b$
- $\nabla {1\over 2}f=\sum_i{(w^Tx_i-y_i)x_i}$
- $\nabla^2 f = x^2$
- We require $\nabla f = \vec{0}$ to solve for $w$
- We want $\nabla^2 f > \vec{0}$ which is always true

### Issues 
- **Overfitting**: We want the error to be small but not zero (not exactly match every data points)
- **Complexity**: Too complex to solve $\nabla f = \vec{0}$ with high dimension $w$