source("https://raw.githubusercontent.com/hyunseungkang/invalidIV/master/JMCode.R")
source("https://raw.githubusercontent.com/hyunseungkang/invalidIV/master/TSHT.R")

D <- as.matrix(read.csv('./treat.csv', header=F))
Y <- as.matrix(read.csv('./response.csv', header=F))
Z <- as.matrix(read.csv('./inst.csv',header=F))
X <- as.matrix(read.csv('./feat.csv',header=F))
n = length(Y)
model <- TSHT(Y,D,Z, X)

D_test <- as.matrix(read.csv('./treat_test.csv', header=F))
Y_test <- read.csv('./response_test.csv', header=F)
Z_test <- as.matrix(read.csv('./inst_test.csv',header=F))
X_test <- as.matrix(read.csv('./feat_test.csv',header=F))

sq_diff<-(Y_test - D_test * model$betaHat[1,1])^2

cat(c(model$betaHat[1,1],  mean(sq_diff$V1)), sep=",", file="outfile.txt", append=TRUE)
cat("\n", file="outfile.txt", append=TRUE)

