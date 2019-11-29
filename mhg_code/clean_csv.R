filesToDelete <- dir()[unlist(lapply(dir(),function(s){endsWith(s,'.csv')}))]

for(f in filesToDelete) {
  file.remove(f)
}

filesToCopy <- dir('./starting_csv')
for(f in filesToCopy) {
  name_ext <- strsplit(f,'[.]')[[1]]
  file.copy(paste0('./starting_csv/',f),paste0(substr(name_ext[1],1,nchar(name_ext[1])-1),'.',name_ext[2]),overwrite=T)
}
