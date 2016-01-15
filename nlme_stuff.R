# figuring out lme4

library(nlme)
library(lme4)
load('RGB.gzip')
load('SPOTS.gzip')
# load("Xdf.gzip")
# load("Ydf.gzip")
# load("id.gzip")
# load("spots.gzip")
# load("time.gzip")
# 9B, 36B, 46G
load("longdata.gzip")

#weights=varIdent(form = ~ 1 | diagnosis.ASD)

# newdf= RGB[which(RGB$time %in% c(1000,1020,1040,1060)),]
# fit = glmer(detection ~time+R+G+B+as.factor(spot)+(1|trial),
#             data=newdf, 
#             family = binomial(logit))
# summary(fit)

df = SPOTS[which(SPOTS$time %in% c(200,600,1200)),
c("time", "X21G", "X23B", "X47R", "trial", "detection")]
fit2 = glmer(detection ~ time + X21G + X23B + X47R + (1|trial),
             data=df,
             family = binomial(logit))
summary(fit2)
fit5 = glmer(detection ~ time + X47R + (1|trial),
             data=df,
             family=binomial, 
             nAGQ=75)
summary(fit5)

load("ECG.gzip")
fit3 = glmer(ECG ~ TRT + Period + (1 | ID), 
      data = ECG,
      family=binomial, 
      nAGQ=75)
summary(fit3)

load("AVG.gzip")
avgdf = AVG[which(AVG$time %in% c(200,600,1200)),
fit6 = glmer(detection ~ time + X47R + (1|trial),
             data=avgdf,
             family=binomial, 
             nAGQ=75)
summary(fit6)