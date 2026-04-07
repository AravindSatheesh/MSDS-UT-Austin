## Problem 1 ##
data = read.csv('HW3_data.csv') 

#check the structure of the data frame
str(data) 
y = data$y
m = data$m
x = data$x
n = length(y)

X = cbind(rep(1, n), x)
MX = cbind(rep(1, n), m, x)

# Compute the estimators
beta_1 = solve(t(X)%*%X) %*% t(X) %*% y
print(beta_1)
beta_2 = solve(t(X)%*%X) %*% t(X) %*% m
print(beta_2)
beta_3 = solve(t(MX)%*%MX) %*% t(MX) %*% y
print(beta_3)

# Compute the variance of beta_22 and beta_32
V_22 = solve(t(X)%*%X)[2, 2]
print(V_22)
V_32 = solve(t(MX)%*%MX)[2, 2]
print(V_32)

# Compute the test statistic
z = (beta_1[2] - beta_3[3]) / sqrt(beta_2[2]^2 * V_32 + beta_3[2]^2 * V_22)
print(z)

# Compute the p-value
p = 2 * (1 - pnorm(z, 0, 1))
print(p)