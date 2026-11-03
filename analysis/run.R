# Run from repository root: Rscript analysis/run.R
suppressPackageStartupMessages({library(glmnet); library(rpart)})
set.seed(20261103)
dir.create('analysis/results',showWarnings=FALSE); out <- 'analysis/results'
writecsv <- function(x,n) write.csv(x,file.path(out,paste0(n,'.csv')),row.names=FALSE)
d <- read.csv('data/add.csv',check.names=FALSE,na.strings=c('?',' ?','  ?','   ?','    ?','     ?'),strip.white=TRUE)
stopifnot(nrow(d)==3279,ncol(d)==1560,identical(d[[1]],0:3278))
y <- as.integer(trimws(d[[ncol(d)]])=='ad.')
x <- as.matrix(d[,2:(ncol(d)-1)]); storage.mode(x)<-'double'
ns <- readLines('analysis/source/ad.names'); nm <- sub(':.*','',grep('^[^|].*:',ns,value=TRUE)); stopifnot(length(nm)==ncol(x)); colnames(x)<-nm
# Group by the entire observed predictor pattern, including missingness, never labels.
key <- apply(x,1,function(z) paste(ifelse(is.na(z),'NA',format(z,digits=15,trim=TRUE)),collapse='|'))
g <- match(key,unique(key)); ug<-unique(g); gy<-sapply(ug,function(k)as.integer(mean(y[g==k])>=.5))
splitg<-rep('',length(ug))
for(v in 0:1){ ids<-sample(which(gy==v)); n<-length(ids); splitg[ids]<-c(rep('train',floor(.6*n)),rep('validation',floor(.2*n)),rep('test',n-floor(.6*n)-floor(.2*n))) }
part<-splitg[g]; tr<-which(part=='train'); va<-which(part=='validation'); te<-which(part=='test')
foldg<-integer(length(ug)); for(v in 0:1){ids<-which(splitg=='train' & gy==v);foldg[ids]<-sample(rep(1:5,length.out=length(ids)))}
fold<-foldg[g]
stopifnot(!any(g[tr]%in%g[te]),!any(g[va]%in%g[te]),!any(g[tr]%in%g[va]))
writecsv(data.frame(row=seq_along(y),original_index=d[[1]],group=g,partition=part,fold=fold,y=y),'splits')
writecsv(data.frame(feature=nm,missing=colSums(is.na(x)),unique=sapply(seq_len(ncol(x)),function(j)length(unique(na.omit(x[,j])))),active=colSums(x==1,na.rm=TRUE)),'dictionary')
writecsv(do.call(rbind,lapply(c('train','validation','test'),function(p)data.frame(partition=p,n=sum(part==p),groups=length(unique(g[part==p])),ad=sum(y[part==p]),nonad=sum(1-y[part==p])))),'partition_counts')
# Log geometry reduces scale asymmetry without deleting valid large values.
z<-x; z[,1:3]<-log1p(z[,1:3]); miss<-is.na(z[,1:4]); z<-cbind(z,miss*1); colnames(z)[(ncol(x)+1):ncol(z)]<-paste0(nm[1:4],'_missing')
prep<-function(a) apply(a,2,median,na.rm=TRUE)
fill<-function(a,m){for(j in which(colSums(is.na(a))>0))a[is.na(a[,j]),j]<-m[j];a}
auc<-function(y,p){a<-sum(y);b<-length(y)-a;if(a*b==0)return(NA_real_);(sum(rank(p)[y==1])-a*(a+1)/2)/(a*b)}
ap<-function(y,p){o<-order(p,decreasing=TRUE); yy<-y[o]; pp<-p[o]; ends<-c(which(diff(pp)!=0),length(pp)); tp<-cumsum(yy)[ends]; sum(diff(c(0,tp))*tp/ends)/sum(y)}
metrics<-function(y,p,t=.5){h<-as.integer(p>=t);tp<-sum(h*y);fp<-sum(h*(1-y));fn<-sum((1-h)*y);tn<-sum((1-h)*(1-y)); c(accuracy=mean(h==y),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,precision=if(tp+fp)tp/(tp+fp) else 0,recall=tp/(tp+fn),specificity=tn/(tn+fp),F1=if(2*tp+fp+fn)2*tp/(2*tp+fp+fn) else 0,ROC_AUC=auc(y,p),AP=ap(y,p),Brier=mean((p-y)^2),TP=tp,FP=fp,FN=fn,TN=tn)}
lams<-10^seq(2,-4,length.out=25); cps<-c(.001,.005,.01,.02,.05)
cv<-list(); models<-c('geometry_ridge','full_ridge','tree')
for(model in models){cols<-if(model=='geometry_ridge')c(1:4,(ncol(x)+1):ncol(z)) else seq_len(ncol(z)); grid<-if(model=='tree')cps else lams
 for(k in 1:5){a<-tr[fold[tr]!=k];b<-tr[fold[tr]==k];m<-prep(z[a,cols,drop=FALSE]);aa<-fill(z[a,cols,drop=FALSE],m);bb<-fill(z[b,cols,drop=FALSE],m)
 if(model!='tree'){fit<-glmnet(aa,y[a],family='binomial',alpha=0,lambda=lams,standardize=TRUE);pred<-predict(fit,bb,s=lams,type='response')} else {pred<-sapply(grid,function(cp){fit<-rpart(y~.,data=data.frame(y=factor(y[a],levels=0:1),aa,check.names=TRUE),method='class',control=rpart.control(cp=cp,minsplit=20,maxdepth=5,xval=0));predict(fit,data.frame(bb,check.names=TRUE),type='prob')[,'1']})}
 for(j in seq_along(grid))cv[[length(cv)+1]]<-data.frame(model=model,fold=k,parameter=grid[j],AP=ap(y[b],pred[,j]))
 }}
cv<-do.call(rbind,cv);writecsv(cv,'cv');agg<-aggregate(AP~model+parameter,cv,mean);writecsv(agg,'cv_summary')
params<-sapply(models,function(mod){a<-agg[agg$model==mod,];a$parameter[which.max(a$AP)]})
# All model/parameter selection uses training CV only. Threshold uses validation only.
valpred<-testpred<-list();fits<-list(); valrows<-list(); selected<-models[which.max(sapply(models,function(mod)max(agg$AP[agg$model==mod])))]
for(model in models){cols<-if(model=='geometry_ridge')c(1:4,(ncol(x)+1):ncol(z)) else seq_len(ncol(z));m<-prep(z[tr,cols,drop=FALSE]);aa<-fill(z[tr,cols,drop=FALSE],m);bb<-fill(z[va,cols,drop=FALSE],m);cc<-fill(z[te,cols,drop=FALSE],m)
 if(model!='tree'){fit<-glmnet(aa,y[tr],family='binomial',alpha=0,lambda=lams);vp<-as.numeric(predict(fit,bb,s=params[model],type='response'));tp<-as.numeric(predict(fit,cc,s=params[model],type='response'))}else{fit<-rpart(y~.,data=data.frame(y=factor(y[tr],levels=0:1),aa,check.names=TRUE),method='class',control=rpart.control(cp=params[model],minsplit=20,maxdepth=5,xval=0));vp<-predict(fit,data.frame(bb,check.names=TRUE),type='prob')[,'1'];tp<-predict(fit,data.frame(cc,check.names=TRUE),type='prob')[,'1']}
 valpred[[model]]<-vp;testpred[[model]]<-tp;fits[[model]]<-list(fit=fit,median=m,columns=cols,parameter=params[model]);valrows[[model]]<-data.frame(model=model,parameter=params[model],t(metrics(y[va],vp)))
}
writecsv(do.call(rbind,valrows),'validation')
vp<-valpred[[selected]]; thresholds<-sort(unique(c(0,vp,1))); f1<-sapply(thresholds,function(t)metrics(y[va],vp,t)['F1']); threshold<-max(thresholds[f1==max(f1)])
# Freeze final model and threshold before generating the test summary.
saveRDS(list(model=selected,threshold=threshold,models=fits,features=nm),'analysis/results/models.rds')
rows<-lapply(models,function(mod)data.frame(model=mod,threshold=.5,t(metrics(y[te],testpred[[mod]]))))
rows[[4]]<-data.frame(model='selected_tuned',threshold=threshold,t(metrics(y[te],testpred[[selected]],threshold)))
rows[[5]]<-data.frame(model='always_nonad',threshold=.5,t(metrics(y[te],rep(mean(y[tr]),length(te)),1)))
res<-do.call(rbind,rows);writecsv(res,'test_metrics')
pr<-data.frame(row=te,group=g[te],y=y[te]);for(mod in models)pr[[mod]]<-testpred[[mod]];pr$selected<-testpred[[selected]];pr$decision<-as.integer(pr$selected>=threshold);writecsv(pr,'test_predictions')
vpr<-data.frame(row=va,y=y[va]);for(mod in models)vpr[[mod]]<-valpred[[mod]];writecsv(vpr,'validation_predictions')
# Cluster bootstrap resamples complete predictor groups; model and threshold remain fixed.
set.seed(20261104);tg<-unique(g[te]);boot<-replicate(1000,{s<-sample(tg,length(tg),replace=TRUE); ii<-unlist(lapply(s,function(k)which(g[te]==k)),use.names=FALSE); a<-metrics(y[te][ii],pr$selected[ii],threshold);b<-metrics(y[te][ii],testpred[['geometry_ridge']][ii],.5);c(a[1:9],delta_AP=a['AP']-b['AP'])})
writecsv(data.frame(metric=rownames(boot),lower=apply(boot,1,quantile,.025,na.rm=TRUE),upper=apply(boot,1,quantile,.975,na.rm=TRUE)),'bootstrap')
# Descriptive pairwise separation only: repeated rows do not justify an independent-row test.
infer <- do.call(rbind, lapply(1:3, function(j) {
  a <- na.omit(x[tr[y[tr] == 1], j])
  b <- na.omit(x[tr[y[tr] == 0], j])
  u <- sum(rank(c(a, b))[seq_along(a)]) - length(a) * (length(a) + 1) / 2
  data.frame(feature=nm[j], ad_n=length(a), nonad_n=length(b),
             ad_median=median(a), nonad_median=median(b), W=u,
             rank_probability=u / (length(a) * length(b)))
}))
writecsv(infer, 'geometry_association')
desc<-do.call(rbind,lapply(0:1,function(v)do.call(rbind,lapply(1:4,function(j){a<-x[y==v,j];data.frame(class=if(v)'ad' else 'nonad',feature=nm[j],n=sum(!is.na(a)),missing=sum(is.na(a)),mean=mean(a,na.rm=TRUE),sd=sd(a,na.rm=TRUE),q25=quantile(a,.25,na.rm=TRUE),median=median(a,na.rm=TRUE),q75=quantile(a,.75,na.rm=TRUE))}))));writecsv(desc,'descriptive')
cal<-cut(pr$selected,breaks=seq(0,1,.2),include.lowest=TRUE);writecsv(aggregate(cbind(predicted=pr$selected,observed=pr$y,n=rep(1,nrow(pr))),list(bin=cal),function(a)c(mean=mean(a),sum=sum(a))),'calibration')
coefm<-fits[['full_ridge']];cf<-as.matrix(coef(coefm$fit,s=coefm$parameter));ord<-order(abs(cf[-1,1]),decreasing=TRUE)[1:15];writecsv(data.frame(feature=rownames(cf)[-1][ord],coefficient=cf[-1,1][ord]),'coefficients')
pdf(file.path(out,'distributions.pdf'),width=8,height=3.6);par(mfrow=c(1,3),mar=c(4,4,2,1));for(j in 1:3)boxplot(log1p(x[,j])~factor(y,labels=c('nonad','ad')),main=nm[j],xlab='Class',ylab='log(1 + observed value)',col=c('#a6cee3','#fb9a99'));dev.off()
pdf(file.path(out,'confusion.pdf'),width=5,height=3.8);par(mar=c(4,4,2,1));tab<-table(factor(pr$y,levels=0:1),factor(pr$decision,levels=0:1));image(1:2,1:2,t(tab[2:1,]),col=colorRampPalette(c('white','#238b45'))(100),axes=FALSE,xlab='Predicted',ylab='Actual',main='Frozen model and validation threshold');axis(1,1:2,c('nonad','ad'));axis(2,1:2,c('ad','nonad'));for(i in 1:2)for(j in 1:2)text(i,j,tab[3-j,i]);dev.off()
conflict<-sum(sapply(ug,function(k)length(unique(y[g==k]))>1));info<-data.frame(item=c('n','predictors','ads','groups','duplicate_rows','conflicting_groups','selected_model','threshold','seed','train_CV_AP'),value=c(nrow(x),ncol(x),sum(y),length(ug),sum(duplicated(g)),conflict,selected,threshold,20261103,max(agg$AP)));writecsv(info,'summary')
stopifnot(all(is.finite(pr$selected)),all(pr$selected>=0 & pr$selected<=1),sum(tab)==length(te),length(unique(nm))==1558)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
print(info);print(res)
