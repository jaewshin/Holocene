# The script ./mhg_code/'!MoralizingGods.R' is a cleaned version of code from:
#
# https://github.com/pesavage/moralizing-gods
#
# The script outputs the text file MoralisingGodsStatus.txt at the point where
# the relevant Nature analysis occurs.
#
# To check the code in ./mhg_code, the data used to generate Extended Data
# Fig. 1 can be downloaded from here:
#
# https://www.nature.com/articles/s41586-019-1043-4#MOESM6
#
# The file 41586_2019_1043_MOESM6_ESM_sheet1.csv contains sheet1 of the
# associated .xlsx file in .csv format.
#
# This script iterates over the 12 NGAs used in the Nature paper for which data
# 41586_2019_1043_MOESM6_ESM.xlsx to ensure that the code in ./mhg_code gives
# identical results. This is done for both MoralisingGods and DoctrinalMode.

rm(list=ls()) # Clear the workspace
natureData <- read.csv('41586_2019_1043_MOESM6_ESM_sheet1.csv',stringsAsFactors=F)
githubData <- read.csv('./mhg_code/data_used_for_nature_analysis.csv',stringsAsFactors=F)

# Subset to only the columns needed and give the columns the same names
natureData <- natureData[,c('NGA','Date..negative.years.BCE..positive...years.CE.','MoralisingGods','DoctrinalMode')]
names(natureData) <- c('NGA','Time','MoralisingGods','DoctrinalMode')
githubData <- githubData[,c('NGA','Time','MoralisingGods','DoctrinalMode')]


NGAs <- unique(natureData$NGA)

for(i in 1:length(NGAs)) {
  # Subset for convenience
  nga <- NGAs[i]
  nat <- natureData[natureData$NGA == nga,]
  git <- githubData[githubData$NGA == nga,]
  # Set NA to -1 so that all works nicely
  nat[is.na(nat)] <- -1
  git[is.na(git)] <- -1
  if(!all.equal(dim(nat),dim(git))) {
    print(nga)
    print('Failed for dimension')
  } else {
    # Dimensions are OK
    if(!all(nat$Time == git$Time)) {
      print(nga)
      print('Failed for Time data')
    }

    if(!all(nat$MoralisingGods == git$MoralisingGods)) {
      print(nga)
      print('Failed for MoralisingGods data')
    }

    if(!all(nat$DoctrinalMode == git$DoctrinalMode)) {
      print(nga)
      print('Failed for DoctrinalMode data')
    }
  }
}

# Moralising God date by first occurence
for(i in 1:length(NGAs)) {
  # Subset for convenience
  nga <- NGAs[i]
  nat <- natureData[natureData$NGA == nga,]
  git <- githubData[githubData$NGA == nga,]
  # Set NA to -1 so that all works nicely
  nat[is.na(nat)] <- -1
  git[is.na(git)] <- -1
  print('--')
  print(nga)
  print(git$Time[min(which(git$MoralisingGods == 1))])
}
