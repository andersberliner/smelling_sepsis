# figuring out lme4

library(lme4)

# load("Xdf.gzip")
# load("Ydf.gzip")
# load("id.gzip")
# load("spots.gzip")
# load("time.gzip")
load("longdata.gzip")
glmer(Ydf ~ Xdf + (1|id), family = binomial, nAGQ = 75)
summary(glmer(Ydf ~ Xdf + (1|id), family = binomial, nAGQ = 75))

