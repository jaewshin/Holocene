calcNegLogLik <- function(theta,x_out,dx_out,x0) {
  numTerms <- length(theta) / 2
  meanV <- calcMeanV(theta[1:numTerms],x_out,x0)
  sigV <- calcSigV(theta[(numTerms+1):(2*numTerms)],x_out,x0)
  logLik <- dnorm(dx_out,meanV,sigV,log=T)
  return(-sum(logLik))
}

calcMeanV <- function(param,x,x0) {
  v0_bar <- param[1]
  v0 <- exp(v0_bar)
  order <- length(param) - 1

  ind_below <- x <= x0
 
  meanV_above <- rep(v0,sum(!ind_below))
  for(n in 1:order) {
    meanV_above <- meanV_above + param[1+n] * (x[!ind_below]-x0)^n
  }
  meanV_below <- v0*exp(x[ind_below] - x0)
  meanV <- rep(NA,length(x))
  meanV[ind_below]  <- meanV_below
  meanV[!ind_below] <- meanV_above
  return(meanV)
}

calcSigV <- function(param,x,x0) {
  x[x <= x0] <- x0
  sigV_bar <- rep(0,length(x))
  for(n in 1:length(param)) {
    sigV_bar <- sigV_bar + param[n] * x^(n-1)
  }
  sigV <- exp(sigV_bar)
  return(sigV)
}

#calcSigV <- function(param,x) {
#  sigV_bar <-calcMeanV(param,x,-Inf)
#  sigV <- exp(sigV_bar)
#  return(sigV)
#}

simSeries <- function(theta,x_out,dx_out,tau_out,j,x0) {
  order <- length(theta)/2 - 1
  lastInd <- which(is.na(dx_out))
  firstInd <- lastInd + 1
  firstInd <- firstInd[1:(length(firstInd)-1)]
  firstInd <- c(1,firstInd)
  n0 <- firstInd[j]
  n1 <- lastInd[j]
  x <- x_out[n0]

  counter <- 1
  for(n in 1:(n1-n0)) {
    xcurrent <- x[n]
    numSteps <- tau_out[firstInd[j] + n - 1] / 100
    for(m in 1:numSteps) {
      meanV <- calcMeanV(theta[1:(order+1)],xcurrent,x0)
      sigV <- calcSigV(theta[(order+2):(2*order+2)],xcurrent,x0)
      xcurrent <- xcurrent + rnorm(1,meanV,sigV)
      #if(xcurrent < x0) {
      #  xcurrent <- x
      #}
      print('--')
      print(counter)
      print(xcurrent)
      print(meanV)
      print(sigV)
      counter <- counter + 1
    }
    x <- c(x,xcurrent)
  }
  return(x)
}
