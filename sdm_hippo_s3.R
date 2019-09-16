# Run SDM (statistical disease mapping) pipeline 
# on ADNI hippocampal surface data (step 3)

# Author: Rongjie Liu (rongjie.liu@rice.edu)
# Last update: 2019-2-28

library(lctools)
mode <- "left"
coord <- read.table(sprintf("./result/%s/coord.txt", mode), header = F)
colnames(coord) <- c("X", "Y")
label <- read.table(sprintf("./result/%s/label.txt", mode), header = F)
subg <- read.table("./data/subg.txt", header = F)
# mci subgroup
label.subg <- label[subg==1,]
dx <- read.table("./data/dx.txt", header = F)
x <- read.table("./data/x.txt", header = F)
x.subg <- x[dx==1,]
x1 <- x.subg[subg==1,]
# male
male.idx <- which(x1$V1==0)
label.male <- label.subg[male.idx,]
# age range
age.male <- x1$V2[male.idx]
y.all <- label.male[age.male>=65,]
y <- colSums(y.all)

dat <- data.frame(cbind(y,coord))
colnames(dat) <- c("obs", "X1", "X2")
gw.model <- gw.glm(obs ~ X1 + X2, "poisson", dat, 10, 
                kernel = 'adaptive', cbind(coord$X,coord$Y))
mu <- exp(gw.model$GGLM_LEst$X.Intercept.+gw.model$GGLM_LEst$X1*coord$X
+gw.model$GGLM_LEst$X2*coord$Y)/dim(y.all)[1]
write.table(mu,sprintf("./result/%s/prob_%s_%s_%s.txt", mode, "ad","f","2"),  row.names = F,
            col.names = F)