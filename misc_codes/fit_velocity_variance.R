library(mixtools)
library(yada)
rm(list=ls())

source('functions_for_fit_velocity_variance.R')

x_out <- read.csv('x_out.csv',header=0) # 414 x 9
dx_out <- read.csv('dx_out.csv',header=0) # 414 x 9
tau_out <- read.csv('tau_out.csv',header=0) # 414 x 9 [time difference to next point]


# dx_out is na at the end of an NGA's sequence
lastInd <- which(is.na(dx_out[,1]))
firstInd <- lastInd + 1
firstInd <- firstInd[1:(length(firstInd)-1)]
firstInd <- c(1,firstInd)

PC = 1
x_out0 <- x_out[,PC]
dx_out0 <- dx_out[,PC]
tau_out0 <- tau_out[,PC]
#startFit <- normalmixEM(x_out[firstInd])
#plot(startFit,which=2)

ind <- !is.na(dx_out0) & tau_out0 == 100
x_out <- x_out0[ind]
dx_out <- dx_out0[ind]
tau_out <- tau_out0[ind]

order <- 1
theta0 <- c(rep(0,order+1),1,rep(0,order))
#logLikVect <- calcLogLik(theta0,x_out0,dx_out0)

x0 <- min(x_out0)
tempVect <- 100 * .75^(0:50)
temper <- saTemper(calcNegLogLik,theta0,tempVect=tempVect,control=list(numCycles=20,verbose=T),x_out=x_out,dx_out=dx_out,x0=x0)

theta <- saTemperBest(temper)
xplot <- seq(min(x_out0),max(x_out0),len=1000)
meanV <- calcMeanV(theta[1:(order+1)],xplot,x0)
sigV <- calcSigV(theta[(order+2):(2*order+2)],xplot,x0)
plot(xplot,sigV,type='l',lwd=3)
plot(xplot,meanV,type='l',lwd=3)

xall_list <- list()
numSim <- 100
for(j in 1:length(firstInd)) {
  xnew <- simSeries(theta,x_out0,dx_out0,tau_out0,j,x0)
  #plot(xnew)
  xall <- xnew
  for(k in 1:(numSim-1)) {
    print(paste(as.character(j),as.character(k)))
    xnew <- simSeries(theta,x_out0,dx_out0,tau_out0,j,x0)
    #points(xnew)
    xall <- c(xall,xnew)
  }
  xall_list[[j]] <- xall
}

hist(unlist(xall_list),100,xlim=c(min(x_out0),max(x_out0)))
