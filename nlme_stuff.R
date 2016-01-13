# figuring out lme4

library(nlme)

# load("Xdf.gzip")
# load("Ydf.gzip")
# load("id.gzip")
# load("spots.gzip")
# load("time.gzip")
load("longdata.gzip")

#weights=varIdent(form = ~ 1 | diagnosis.ASD)
fit = glmer(detection ~ DI + (1|trial), 
            corr=corAR1(,form= ~ 1 | trial),
            data=longdata, family = binomial, nAGQ = 75)
summary(fit)

