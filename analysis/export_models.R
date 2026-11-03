# Extract the fitted objects used in the report; does not refit models.
suppressPackageStartupMessages(library(glmnet))
s <- readRDS('analysis/results/models.rds')
geometry <- s$models$geometry_ridge
b <- as.matrix(coef(geometry$fit, s=geometry$parameter))
write.csv(data.frame(term=rownames(b), coefficient=b[,1]),
          'analysis/results/geometry_coefficients.csv', row.names=FALSE)
d <- read.csv('data/add.csv', check.names=FALSE,
              na.strings=c('?', ' ?', '  ?', '   ?', '    ?', '     ?'), strip.white=TRUE)
z <- as.matrix(d[,2:(ncol(d)-1)])
z[,1:3] <- log1p(z[,1:3])
z <- cbind(z, is.na(z[,1:4])*1)
# Row 844 is the previously reported repeated false-positive signature, not a new selection.
worked <- do.call(rbind, lapply(c('geometry_ridge','full_ridge'), function(name) {
  m <- s$models[[name]]
  a <- z[844,m$columns,drop=FALSE]
  for (j in which(is.na(a))) a[j] <- m$median[j]
  cf <- as.matrix(coef(m$fit,s=m$parameter))
  eta <- cf[1,1] + sum(a*cf[-1,1])
  data.frame(model=name, row=844, logit=eta, probability=plogis(eta))
}))
write.csv(worked, 'analysis/results/worked_prediction.csv', row.names=FALSE)
write.csv(data.frame(feature=names(geometry$median), median=geometry$median),
          'analysis/results/imputation.csv',row.names=FALSE)
writeLines(capture.output(print(s$models$tree$fit)), 'analysis/results/tree.txt')
pdf('analysis/results/tree.pdf',width=8,height=4.8)
plot(s$models$tree$fit,uniform=TRUE,margin=.12)
text(s$models$tree$fit,use.n=TRUE,all=TRUE,pretty=0,cex=.95)
title('Fitted comparator: splits and training leaf counts')
dev.off()
