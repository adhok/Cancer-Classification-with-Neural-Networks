require(neuralnet)
require(caret)
require(irr)
cancer <- read.table('cancer.txt',sep=',')
cancer$V2 <- ifelse(cancer$V2=='M',1,0)
set.seed(2)
train<- sample(dim(cancer)[1],dim(cancer)[1]*0.9)
cancer_train <- cancer[train,]
cancer_test <- cancer[-train,]
nn <- neuralnet(V2~V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32,data=cancer_train,hidden=2,err.fct='ce',linear.output = FALSE)
png('nn.png')
plot(nn,rep=1)
dev.off()
new.output <- compute(nn,covariate = cancer_test[,3:32])
outputvspred <- data.frame(new.output$net.result,cancer_test$V2)
outputvspred$new.output.net.result<- ifelse(outputvspred$new.output.net.result>0.5,1,0)
class_rp<-mean(outputvspred$new.output.net.result==outputvspred$cancer_test.V2)


###################################################
nn.rp.m <- neuralnet(formula = V2~V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32, data = cancer_train , hidden = 2 ,learningrate = 0.01, algorithm = 'rprop-', err.fct = 'ce', linear.output = FALSE,rep=5)
png('nn_rpm.png')
plot(nn.rp.m,rep=1)
dev.off()
new.output.rpm <- compute(nn.rp.m,covariate = cancer_test[,3:32])
outputvspred.rpm <- data.frame(new.output.rpm$net.result,cancer_test$V2)
outputvspred.rpm$new.output.rpm.net.result<- ifelse(outputvspred.rpm$new.output.rpm.net.result>0.5,1,0)
class_rpm <- mean(outputvspred.rpm$new.output.rpm.net.result==outputvspred.rpm$cancer_test.V2)
#############################################################################################
nn.sag <- neuralnet(formula = V2~V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32, data = cancer_train , hidden = 2 ,learningrate = 0.01, algorithm = 'sag', err.fct = 'ce', linear.output = FALSE,rep=5)
png('nn_sag.png')
plot(nn.sag,rep=1)
dev.off()
new.output.sag <- compute(nn.sag,covariate = cancer_test[,3:32])
outputvspred.sag <- data.frame(new.output.sag$net.result,cancer_test$V2)
outputvspred.sag$new.output.sag.net.result<- ifelse(outputvspred.sag$new.output.sag.net.result>0.5,1,0)
class_sag<-mean(outputvspred.sag$new.output.sag.net.result==outputvspred.sag$cancer_test.V2)
########################################################################################################
nn.slr <- neuralnet(formula = V2~V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32, data = cancer_train , hidden = 2 ,learningrate = 0.01, algorithm = 'slr', err.fct = 'ce', linear.output = FALSE,rep=5)
png('nn_slr.png')
plot(nn.slr,rep=1)
dev.off()
new.output.slr <- compute(nn.slr,covariate = cancer_test[,3:32])
outputvspred.slr <- data.frame(new.output.slr$net.result,cancer_test$V2)
outputvspred.slr$new.output.slr.net.result<- ifelse(outputvspred.slr$new.output.slr.net.result>0.5,1,0)
class_slr<-mean(outputvspred.slr$new.output.slr.net.result==outputvspred.slr$cancer_test.V2)
#############################################################################################
names<-c('rprop+','rprop-','sag','slr')
value <- c(class_rp,class_rpm,class_sag,class_slr)
df_class<- data.frame(names,value)
png('plot_nn.png')
barplot(df_class$value , names.arg=df_class$names,width=1,cex.names=0.5,main=paste('Classification Accuracy'))
dev.off()
# k fold cross validation
set.seed(123)
folds <- createFolds(cancer$V2,5)
cv_results <-lapply(folds,function(x){
  cancer_train <- cancer[x,]
  cancer_test <- cancer[-x,]
  nn <- neuralnet(V2~V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32,data=cancer_train,hidden=2,err.fct='ce',linear.output = FALSE)
  new.output <- compute(nn,covariate = cancer_test[,3:32])
  new.output$net.result<-ifelse(new.output$net.result>0.5,1,0)
  class<- mean(new.output$net.result==cancer_test$V2)
  return(class)
})
cv_results







