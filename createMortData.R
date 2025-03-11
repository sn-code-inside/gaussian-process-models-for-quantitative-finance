createMortData = function(year_start,year_end,age_start,age_end,sex,sex_cat){
  
  # function to import multiple data files 
  read.multiple.files = function(path,pattern,DEL){
    # note: DEL: "death"/"exposure"/"lifetable"
    list.filenames = list.files(path,pattern)
    
    list.data = list()
    
    for (i in 1:length(list.filenames)){
      country = sub(paste(DEL,"-",sep=""),"",list.filenames[i])
      country = sub("-.*","",country)
      current.data = read.table(paste(path,list.filenames[i],
                                      sep="/"),header=TRUE)
      current.data$Country = country
      list.data[[i]] = current.data
    }
    out = do.call(rbind,list.data)
    
    return(out)
  }
  
  # import death text files:
  D = read.multiple.files("./data", pattern="death[[:graph:]]+.txt", DEL = "death")
  D = data.table::as.data.table(D)
  
  # import exposure text files:
  E = read.multiple.files("./data", pattern="exposure[[:graph:]]+.txt", DEL = "exposure")
  E = data.table::as.data.table(E)
  
  # convert Age in both files to numeric format:
  D$Age = as.character(D$Age); D = D[Age!="110+"]; D$Age = as.numeric(D$Age)
  E$Age = as.character(E$Age); E = E[Age!="110+"]; E$Age = as.numeric(E$Age)
  
  # if sex_cat="yes" then sex is added as categorical variable:
  if (sex_cat=="yes"){
    Dl = melt(D,id.vars=c("Year","Age","Country","Total")); El = melt(E,id.vars=c("Year","Age","Country","Total"))
    Dl = Dl[Year %in% year_start:year_end & Age %in% age_start:age_end]
    El = El[Year %in% year_start:year_end & Age %in% age_start:age_end]
    rate = Dl$value/El$value
    dat = cbind(Dl[,.(Year,Age,Country,variable)],rate)
    dat = dat[,y:=log(rate)]
    names(dat)[names(dat)=="variable"]="sex"
    dat = dat[order(Country,Year,Age,sex),.(Year,Age,Country,sex,rate,y)]
    levels(dat$sex) = c("0","1"); dat$sex=as.numeric(dat$sex)-1
    # Female is coded as 0 and Male is code as 1
    names(dat) = tolower(names(dat))
  }
  else if (sex_cat=="no"){
    if (sex=="female"|sex=="Female"|sex=="F"|sex=="f"){
      D = D[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Female,Country)]
      E = E[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Female,Country)]
      rate = D$Female/E$Female}
    
    else if (sex=="male"|sex=="Male"|sex=="M"|sex=="m"){
      D = D[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Male,Country)]
      E = E[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Male,Country)]
      rate = D$Male/E$Male}
    
    else {
      D = D[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Total,Country)]
      E = E[Year %in% year_start:year_end & Age %in% age_start:age_end,.(Year,Age,Total,Country)]
      rate = D$Total/E$Total}
    
    dat = cbind(D[,.(Year,Age,Country)],rate)
    dat = dat[,y:=log(rate)]
    dat = dat[order(Country,Age,Year),.(Age,Year,Country,rate,y)]
    names(dat) = tolower(names(dat))
  }
  else stop ("sex_cat must be either yes or no")
  
  return(dat)
}
