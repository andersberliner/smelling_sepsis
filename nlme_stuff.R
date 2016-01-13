# figuring out lme4

library(nlme)
library(lme4)

# load("Xdf.gzip")
# load("Ydf.gzip")
# load("id.gzip")
# load("spots.gzip")
# load("time.gzip")
load("longdata.gzip")

#weights=varIdent(form = ~ 1 | diagnosis.ASD)

newdf= longdata[which(longdata$time %in% c(220,240,260,280,300)), c('detection', 'DI')]
fit = glmer(detection ~time+DI+color_G+color_R+as.factor(spot)+(1|trial),
            data=newdf, 
            family = binomial(logit))
summary(fit)

# predict(model, newdata=new.cars)

