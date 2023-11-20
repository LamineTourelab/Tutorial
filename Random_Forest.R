#######################################################################################################
############################ Analyse lipidomique des foies de souris  #################################
#######################################################################################################

setRepositories()### permet de définir le lieu où sont entreposés les packages
library("beeswarm")
library("reshape2")
library("ggplot2")
library("party")
library("cluster")
library("class")
library("clValid")
library("gplots")
library("gtools")
library("RColorBrewer")
library("pvclust")
require("pvclust")
require("rpart")
library("FactoMineR")
require("FactoMineR")
library("supclust")
require("supclust")
library("multtest")
library("randomForest")
require("randomForest")
library("varSelRF")
require("varSelRF")
library("ROCR")
library("pROC")
library("AUCRF")
library("languageR")
library("designGG")
library("party") 
library("Hmisc")
library("heatmap3")
library("pamr")
update.packages(checkBuilt=TRUE, ask=FALSE)####update packages

rm(list=ls())### permet d'éliminer les éléments uploadés lors d'une analyse précédente= session précédente
ls()### verification que la liste est vide ou dan RStudio que la fenêtre "Envoronnement est vide#


#############################################################################################
################  Analyse de la liste de lipides extrait de l'analyse pamr ci-dessus  #######
#############################################################################################


#############################################################################################
####################  l'analyse du 28/08/2015 apres pamr à sélectionner 38 lipides
#################### le but de l'analyse RF qui suit est d'utiliser le fichier non normaliser "dataframesouriswork1"
#################### au lieu du fichier extrat de l'analyse pam "resultfoie2"

###dataframesouriswork1 <- read.csv2("C:/Users/Franck/Desktop/dataframesouriswork1.csv", header=T, dec=".")

dataframesouriswork1 <- read.csv2("C:/Users/33689/Desktop/dataframesouriswork1.csv", header=T, dec=".")
attach(dataframesouriswork1)
View(dataframesouriswork1)


########################################################################################
########### dans le dossier "dataframesouriswork1" il faut sélectionner que le foie  ###

dataframe <-subset(dataframesouriswork1, Tissue=="liver")
View(dataframe)


###########  analyse des 38 lipides trouvés dans le foie après pamr analyse pour threshold 2.301  #########
usedVar =cbind("Cholesterol","CholC16","CholC18","DG1618","C51TG161616","C53TG161618","C55TG161818","C57TG181818","FAME_140","FAME_160","FAME_180","FAME_200","FAME_161w7","FAME_181w7","FAME_181w9","FAME_201w9","FAME_182w6","FAME_202w6","FAME_183w3","FAME_183w6","FAME_204w6","FAME_225w3","EICO_12HETE","EICO_18HEPE","PC160160","PC160161","PC160181","PC160182","PC161182","PC180181","PC181181","PC181182","PC182182","PC182204","PI181181","PI181182","PE180226","PE181205")

############# boxplots correspondat à chaque variable   ##########
library("reshape2")
lipid <- dataframe[,c("groupe",usedVar)]#### remplacer groups par groupe>>>> qu'est que cela donne???
df.m <- melt(lipid, id.var = "groupe")

x11()
p <- ggplot(data = df.m, aes(x=variable, y=value)) +
  geom_boxplot(aes(fill=groupe))
p + facet_wrap( ~ variable, scales="free")

library(beeswarm)
boxplot(Cholesterol~groups, data=dataframe)
beeswarm(Cholesterol~groups, add=T, data= dataframe)


lipids <- dataframe[,usedVar]
rownames(lipids) <- dataframe$group
x11()
par(mfrow=c(1,2))
dv <- diana(lipids, metric = "manathan", stand = TRUE)
plot(dv)


z <- t(as.matrix(lipids))
rc <- rainbow(nrow(z), start=0, end=.3)
cc <- rainbow(ncol(z), start=0, end=.3)
is.matrix(z)
x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(z, Colv=as.dendrogram(dv), col = greenred(256), scale="row", margins=c(5,10), density="density", xlab = "Patients", ylab= "Lipids", main = "heatmap(Lipidomic souris foie lipides pamr)",breaks=seq(-1.5,1.5,length.out=257))



View(z)
x11()
lipid.pv <- pvclust(z, nboot=1000)
plot(lipid.pv)
ask.bak <- par()$ask
par(ask=TRUE)
pvrect(lipid.pv)
print(lipid.pv, digits=4)###digits = nbre de chiffres après la virgule
print(lipid.pv, digits=3)
x11()
msplot(lipid.pv, edges=c(2,4,6,7))
par(ask=ask.bak)
lipid.pp <- pvpick(lipid.pv)
lipid.pp
summary(lipid.pv)
lipid.pv$msfit
lipid.pv$count
lipid.pv$edges
lipid.pv$hclust

x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(z, Colv=as.dendrogram(lipid.pv$hclust), col = greenred(256), scale="row", margins=c(5,10), density="density", xlab = "Patients", ylab= "Lipids", main = "heatmap(Lipidomic souris foie lipides pamr)",breaks=seq(-1.5,1.5,length.out=257))


#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa <- PCA(lipids, graph=T)
sourisfoie <- HCPC(res.pcaa, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)

lipidsss <- data.frame(groupsssss=dataframe$groups,dataframe[, c(usedVar)])
lipidss <- data.frame(groupsssss=dataframe$groups,lipids)
#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa <- PCA(lipidss, scale.unit=TRUE, ncp=40, quali.sup=c(1:1), graph = FALSE)
sourisfoie <- HCPC(res.pcaa, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)
x11()
plotellipses(res.pcaa)

dimdesc(res.pcaa)$Dim.2$quanti[,1]
dimdesc(res.pcaa)$Dim.3$quanti[,1]
#############################################################################################################
################################  ANALYSE EN RANDOM FOREST   ################################################
#############################################################################################################


##### used the "usedVar =cbind()" above

z1 <- as.matrix(lipids)


### détermine le meilleur "mtry"
bestmtry <- tuneRF(z1, group, ntreeTry=12, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=T)
### mtry=12


lipid.rf <- randomForest(z1, data=dataframe, importance=TRUE, proximity=TRUE, mtry=12, ntree=10000, keep.forest=TRUE)
x11()
varImpPlot(lipid.rf)

####################  MDSplot=Multi-dimensional Scaling Plot of Proximity matrix from randomForest #######
##### Tracer les coordonnées d'échelle de la matrice de proximité de Forest aléatoire ####################
### "PCA" des "groupe"= "Control", "NAFLD" et "NASH" en fonction de l'analyse Randome Forest "lipid.rf ###
##########################################################################################################

x11()
MDSplot(lipid.rf, dataframe$groups, palette=rep(1, 3), pch=as.numeric(dataframe$groups))   ### "groups" correspoind à la colonne des 3 groupe "control", HFD et MCD
legend(x=0, y=0, legend=c("Control", "HFD", "MCD"), pch= c(1:3))

################# alternative pour faire un RF "on regarde les branche pas les feuilles ############
#### dans ce cas on n'utilise pas la matrice mais un "vecteur=B" ou il y a les groupes  ############
##### groupe= "Control", "NAFL1", "NAFL2", "NAFL3", "NASH" qui représentent les "branches" #########
###### c'est pour ça que le "MDS" qui en découle est plus jolie ####################################



B <- group~Cholesterol+CholC16+CholC18+DG1618+C51TG161616+C53TG161618+C55TG161818+C57TG181818+FAME_140+FAME_160+FAME_180+FAME_200+FAME_161w7+FAME_181w7+FAME_181w9+FAME_201w9+FAME_182w6+FAME_202w6+FAME_183w3+FAME_183w6+FAME_204w6+FAME_225w3+EICO_12HETE+EICO_18HEPE+PC160160+PC160161+PC160181+PC160182+PC161182+PC180181+PC181181+PC181182+PC182182+PC182204+PI181181+PI181182+PE180226+PE181205
### B est ce que l'on appelle une formule dans R

lipid.rfalt <- randomForest(B, data=dataframe, importance=TRUE, proximity=TRUE, ntree=1000, mtry=12, keep.forest=TRUE)
x11()
varImpPlot(lipid.rfalt)



lipid.rfalt <- randomForest(B, data=dataframe, importance=TRUE, proximity=TRUE, ntree=1000, mtry=12, keep.forest=TRUE)
x11()
varImpPlot(lipid.rfalt)

x11()
MDSplot(lipid.rfalt, dataframe$groups, palette=rep(1, 3), pch=as.numeric(dataframe$groups))
legend(x=-0.4, y=0.4, legend=c("Control", "HFD", "MCD"), pch= c(1:3))



#### important "groups" contient les 5 groupes Control", "NAFL1", "NAFL2", "NAFL3", "NASH", si on utilise les patients individuel cela ne marche pas
#### le randome forest correspond donc aux lipides qui vont disciminer au mieux les 5 goupes ########################################
#### la fonction "cforest" n'accepte pas les matrice



B <- groups~Cholesterol+CholC16+CholC18+DG1618+C51TG161616+C53TG161618+C55TG161818+C57TG181818+FAME_140+FAME_160+FAME_180+FAME_200+FAME_161w7+FAME_181w7+FAME_181w9+FAME_201w9+FAME_182w6+FAME_202w6+FAME_183w3+FAME_183w6+FAME_204w6+FAME_225w3+EICO_12HETE+EICO_18HEPE+PC160160+PC160161+PC160181+PC160182+PC161182+PC180181+PC181181+PC181182+PC182182+PC182204+PI181181+PI181182+PE180226+PE181205

data.controls <- cforest_unbiased(ntree = 1700, mtry=8)
data.controls
data.cforest <- cforest(B, data= dataframe, controls=data.controls)
data.cforest.varimp <- varimp(data.cforest, conditional = TRUE) 
data.cforest.varimp

x11()
dotplot(sort(data.cforest.varimp), xlab = "Variable Importance",
panel = function(x,y)
{panel.dotplot(x, y, col="darkblue", pch=16, cex=1.1)
panel.abline(v=abs(min(data.cforest.varimp)), col="red", lty="longdash", lwd=2)
panel.abline(v=0, col="blue")})

x11()
barplot(sort(data.cforest.varimp), horiz=TRUE, xlab="Variable Importance in mydata", las=1, ylbias=1)
abline(v=abs(min(data.cforest.varimp)), col="red", lty="longdash", lwd=2) 

########## Cette approche permet de sélectionner le plus petit nombre de variables 
########## capablent de discriminer à la foies les branches et les feuilles
########## ce qui permet de déterminer du coup le OOB: Out of bag error rate.




########## Cette approche permet de sélectionner le plus petit nombre de variables 
########## capablent de discriminer à la foies les branches et les feuilles
########## ce qui permet de déterminer du coup le OOB: Out of bag error rate.

z1  <- as.matrix(lipids) ##### on reprend la matrice créee au début
cl <-dataframe$groups ###### correspond aux "branches" = les 5 groupes   "Control", "NAFL1", "NAFL2", "NAFL3", "NASH"
rf.vs1 <- varSelRF(z1, cl, ntree = ncol(z1)*10, ntreeIterat = 2000, vars.drop.frac = 0.2,  mtryFactor = 25, whole.range = TRUE, recompute.var.imp = FALSE, returnFirstForest = TRUE, keep.forest = TRUE)
rf.vs1
lipids.selection <- rf.vs1$selected.vars     ########selection des lipides pour un furture heatmap
lipids.selection.heatmap <- rf.vs1$selected.model    ###### selection des lipides pour un furture heat map sous la forme "lipid"1" + "lipid2"
rf.vs1$initialImportance
rf.vs1$firstForest
x11()
par(mfrow=c(1,2))
plot(rf.vs1)
rf.vsb <- varSelRFBoot(z1, cl, bootnumber = 20, usingCluster = FALSE, srf = rf.vs1)
rf.vsb
x11()
par(mfrow=c(3,2))
summary(rf.vsb)
plot(rf.vsb) ### Plots of out-of-bag predictions and OOB error vs. number of variables.


##########Variable importances from random forest on permuted class lab "randomVarImpsRF "
#lipid.rf <- randomForest(z1, data= dataframe, ntree = ncol(z1)*10, mtry=10, do.trace = 1000, importance=TRUE, proximity=TRUE, keep.forest=TRUE)
rf.rvi <- randomVarImpsRF(z1, cl, lipid.rf, numrandom = 100, usingCluster = FALSE)
x11()
randomVarImpsRFplot(rf.rvi, lipid.rf, cexPoint = 1, show.var.names = TRUE)
############Selection probability plot for variable importance from random forests "selProbPlot"
x11()
selProbPlot(rf.vsb, k= c(9,20), legend = TRUE, xlegend = 15, ylegend =0.8)





####  lipides majeur après Randome Forest (voir le graphique Variable Importnace"
usedVarInter =cbind("Cholesterol","CholC16","CholC18","FAME_140","FAME_200","FAME_161w7","FAME_181w7","FAME_225w3","EICO_12HETE","EICO_18HEPE","PC160160","PC160161","PC160181","PC160182","PC161182","PC182182","PC182204","PI181181","PI181182","PE180226","PE181205")


############# boxplots correspondat à chaque variable   ##########
##### library("reshape2")
lipidInter1 <- dataframe[,c("groups",usedVarInter)]
df.m <- melt(lipidInter1, id.var = "groups")
x11()
p <- ggplot(data = df.m, aes(x=variable, y=value)) +
  geom_boxplot(aes(fill=groups))
p + facet_wrap( ~ variable, scales="free")


lipidInter <- dataframe[,usedVarInter]
rownames(lipidInter) <- dataframe$group
par(mfrow=c(1,2))
dv32<- diana(lipidInter, metric = "manathan", stand = TRUE)
plot(dv32)

z32 <- t(as.matrix(lipidInter))
rc <- rainbow(nrow(z32), start=0, end=.3)
cc <- rainbow(ncol(z32), start=0, end=.3)
is.matrix(z32)
x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(z32, Colv=as.dendrogram(dv32), col = greenred(256), scale="row", margins=c(5,10), density="density", xlab = "Patients", ylab= "Lipids", main = "heatmap(Lipidomic souris foie lipides pamr)",breaks=seq(-1.5,1.5,length.out=257))



#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa32 <- PCA(lipidInter, graph=T)
sourisfoie32 <- HCPC(res.pcaa32, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)

lipids32 <- data.frame(groupe=dataframe$groups,dataframe[, c(usedVarInter)])
lipidss32 <- data.frame(groupe=dataframe$groups,lipidInter)
#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa32 <- PCA(lipidss32, scale.unit=TRUE, ncp=40, quali.sup=c(1:1), graph = FALSE)
sourisfoie32 <- HCPC(res.pcaa32, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)
dimdesc(res.pcaa32) ###permet de définir les dimension de la PCA pour lesquels les lipides son discriminant
dimdesc(res.pcaa32, axes=1:5) ###on regarde les dimension de la PCA de 1 à 5
x11()
plotellipses(res.pcaa32, axis=c(1,2)) ### on représente la PCA sur les dimension significative ici 1 et 3 d'après le resultat dimdesc

dimdesc(res.pcaa32)$Dim.1$quanti[,1]
dimdesc(res.pcaa32)$Dim.2$quanti[,1]
dimdesc(res.pcaa32)$Dim.3$quanti[,1]


##################################################################################################
##################################################################################################

############### 15 lipides commun foie et serum à partir des 38 lipides foie et 38 lipides serum

####################################################################################################


usedVarInter15 =cbind("Cholesterol","CholC16","CholC18","FAME_140","FAME_160","FAME_180","FAME_161w7","FAME_202w6","FAME_183w3","FAME_183w6","FAME_204w6","FAME_225w3","PC180181","PC181181","PC181182")


############# boxplots correspondat à chaque variable   ##########
##### library("reshape2")
lipidInter15 <- dataframe[,c("groups",usedVarInter15)]
df.m15 <- melt(lipidInter15, id.var = "groups")
x11()
p <- ggplot(data = df.m15, aes(x=variable, y=value)) +
  geom_boxplot(aes(fill=groups))
p + facet_wrap( ~ variable, scales="free")


lipidInter15 <- dataframe[,usedVarInter15]
rownames(lipidInter15) <- dataframe$group
x11()
par(mfrow=c(1,2))
dv15<- diana(lipidInter15, metric = "manathan", stand = TRUE)
plot(dv15)

z15 <- t(as.matrix(lipidInter15))
rc <- rainbow(nrow(z15), start=0, end=.3)
cc <- rainbow(ncol(z15), start=0, end=.3)
is.matrix(z37)
x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(z15, Colv=as.dendrogram(dv15), col = greenred(256), scale="row", margins=c(5,10), density="density", xlab = "Patients", ylab= "Lipids", main = "heatmap(Lipidomic souris foie lipides pamr)",breaks=seq(-1.5,1.5,length.out=257))



#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa15 <- PCA(lipidInter15, graph=T)
sourisfoie15 <- HCPC(res.pcaa15, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)

lipids15 <- data.frame(groupe=dataframe$groups,dataframe[, c(usedVarInter15)])
lipidss15 <- data.frame(groupe=dataframe$groups,lipidInter15)
#### analyse en composante principal des résultats de "lipid" resultat de pamr
### 
res.pcaa15 <- PCA(lipidss15, scale.unit=TRUE, ncp=40, quali.sup=c(1:1), graph = FALSE)
sourisfoie15 <- HCPC(res.pcaa15, nb.clust=3, cluster.CA="rows",order=TRUE,nb.par=3,proba=0.05,consol=TRUE, iter.max=10)
dimdesc(res.pcaa15) ###permet de définir les dimension de la PCA pour lesquels les lipides son discriminant
dimdesc(res.pcaa15, axes=1:5) ###on regarde les dimension de la PCA de 1 à 5
x11()
plotellipses(res.pcaa15, axis=c(1,2)) ### on représente la PCA sur les dimension significative ici 1 et 3 d'après le resultat dimdesc

dimdesc(res.pcaa15)$Dim.1$quanti[,1]
dimdesc(res.pcaa15)$Dim.2$quanti[,1]
dimdesc(res.pcaa15)$Dim.3$quanti[,1]



detach(dataframesouriswork1)

########################################################################################################################
###################    Correlation des 21 lipides trouver dans le foie avec le serum    #################################
##################   21 lipides >>>>  usedVarInter =cbind("Cholesterol","CholC16","CholC18","FAME_140","FAME_200","FAME_161w7","FAME_181w7","FAME_225w3","EICO_12HETE","EICO_18HEPE","PC160160","PC160161","PC160181","PC160182","PC161182","PC182182","PC182204","PI181181","PI181182","PE180226","PE181205")
########################################################################################################################

dataframesouriswork1 <- read.csv2("C:/Users/Franck/Desktop/dataframesouriswork1.csv", header=T, dec=".")
attach(dataframesouriswork1)


x11()
par(mfrow=c(5,3))
plot(Cholesterol[Tissue=="liver"][groups=="Control"],Cholesterol[Tissue=="serum"][groups=="Control"], ylim=c(300,1700))
model <- lm(Cholesterol[Tissue=="serum"][groups=="Control"]~Cholesterol[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(Cholesterol[Tissue=="liver"][groups=="HFD"],Cholesterol[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(300,1700))
model1 <- lm(Cholesterol[Tissue=="serum"][groups=="HFD"]~Cholesterol[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(Cholesterol[Tissue=="liver"][groups=="MCD"],Cholesterol[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(300,1700))
model2 <- lm(Cholesterol[Tissue=="serum"][groups=="MCD"]~Cholesterol[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)


plot(CholC16[Tissue=="liver"][groups=="Control"],CholC16[Tissue=="serum"][groups=="Control"], ylim=c(0,300))
model <- lm(CholC16[Tissue=="serum"][groups=="Control"]~CholC16[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(CholC16[Tissue=="liver"][groups=="HFD"],CholC16[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,300))
model1 <- lm(CholC16[Tissue=="serum"][groups=="HFD"]~CholC16[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(CholC16[Tissue=="liver"][groups=="MCD"],CholC16[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,300))
model2 <- lm(CholC16[Tissue=="serum"][groups=="MCD"]~CholC16[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)



plot(CholC18[Tissue=="liver"][groups=="Control"],CholC18[Tissue=="serum"][groups=="Control"], ylim=c(0,2500))
model <- lm(CholC18[Tissue=="serum"][groups=="Control"]~CholC18[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(CholC18[Tissue=="liver"][groups=="HFD"],CholC18[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,2500))
model1 <- lm(CholC18[Tissue=="serum"][groups=="HFD"]~CholC18[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(CholC18[Tissue=="liver"][groups=="MCD"],CholC18[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,2500))
model2 <- lm(CholC18[Tissue=="serum"][groups=="MCD"]~CholC18[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)

plot(FAME_140[Tissue=="liver"][groups=="Control"],FAME_140[Tissue=="serum"][groups=="Control"], ylim=c(0,30))
model <- lm(FAME_140[Tissue=="serum"][groups=="Control"]~FAME_140[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(FAME_140[Tissue=="liver"][groups=="HFD"],FAME_140[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,30))
model1 <- lm(FAME_140[Tissue=="serum"][groups=="HFD"]~FAME_140[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(FAME_140[Tissue=="liver"][groups=="MCD"],FAME_140[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,30))
model2 <- lm(FAME_140[Tissue=="serum"][groups=="MCD"]~FAME_140[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)

plot(FAME_200[Tissue=="liver"][groups=="Control"],FAME_200[Tissue=="serum"][groups=="Control"], ylim=c(0,120))
model <- lm(FAME_200[Tissue=="serum"][groups=="Control"]~FAME_200[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(FAME_200[Tissue=="liver"][groups=="HFD"],FAME_200[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,120))
model1 <- lm(FAME_200[Tissue=="serum"][groups=="HFD"]~FAME_200[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(FAME_200[Tissue=="liver"][groups=="MCD"],FAME_200[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,120))
model2 <- lm(FAME_200[Tissue=="serum"][groups=="MCD"]~FAME_200[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)



x11()
par(mfrow=c(5,3))
plot(FAME_161w7[Tissue=="liver"][groups=="Control"],FAME_161w7[Tissue=="serum"][groups=="Control"], ylim=c(0,50))
model <- lm(FAME_161w7[Tissue=="serum"][groups=="Control"]~FAME_161w7[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(FAME_161w7[Tissue=="liver"][groups=="HFD"],FAME_161w7[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,50))
model1 <- lm(FAME_161w7[Tissue=="serum"][groups=="HFD"]~FAME_161w7[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(FAME_161w7[Tissue=="liver"][groups=="MCD"],FAME_161w7[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,50))
model2 <- lm(FAME_161w7[Tissue=="serum"][groups=="MCD"]~FAME_161w7[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)


plot(FAME_181w7[Tissue=="liver"][groups=="Control"],FAME_181w7[Tissue=="serum"][groups=="Control"], ylim=c(0,350))
model <- lm(FAME_181w7[Tissue=="serum"][groups=="Control"]~FAME_181w7[Tissue=="liver"][groups=="Control"])
abline(model, col="black")
plot(FAME_181w7[Tissue=="liver"][groups=="HFD"],FAME_181w7[Tissue=="serum"][groups=="HFD"], type="p", col="red", ylim=c(0,350))
model1 <- lm(FAME_181w7[Tissue=="serum"][groups=="HFD"]~FAME_161w7[Tissue=="liver"][groups=="HFD"])
abline(model1, col="red")
plot(FAME_181w7[Tissue=="liver"][groups=="MCD"],FAME_181w7[Tissue=="serum"][groups=="MCD"], type="p", col="green", ylim=c(0,350))
model2 <- lm(FAME_181w7[Tissue=="serum"][groups=="MCD"]~FAME_181w7[Tissue=="liver"][groups=="MCD"])
abline(model2, col="green")

model
model1
model2
summary(model)[[8]]#### permet d'extraire le R2 de la fonction linéaire
summary(model1)[[8]]
summary(model2)[[8]]
anova(model)#### permet de tester R2 de la fonction linéaire pour voir p
anova(model1)
anova(model2)

### etc..... avec toutes les varible à tester 

########################################################################################################################
###################    Correlation des 21 lipides trouver dans le foie avec le serum    #################################
##################   11 lipides de la PCA discriminant les MCD
###############    > dimdesc(res.pcaa32)$Dim.2$quanti[,1]
############### PC182182    PC182204    PC160182    PC161182    PE181205 EICO_12HETE Cholesterol 
############### 0.9401747   0.9247872   0.8915219   0.8168413   0.7979749   0.6931409   0.6929198 
##############  EICO_18HEPE    PC160160    PI181181    PE180226 
##############  0.6837172   0.6541528   0.6275945   0.5602314 
#########################################################################################





####  11 lipides sur les 21 de la PCA dim$2 permettant de disciminer les MCD
usedVarInter =cbind("Cholesterol","EICO_12HETE","EICO_18HEPE","PC160160","PC160182","PC161182","PC182182","PC182204","PE181205","PE180226","PI181181")


############# boxplots correspondat à chaque variable   ##########
##### library("reshape2")
lipidInter1 <- dataframe[,c("groups",usedVarInter)]
df.m <- melt(lipidInter1, id.var = "groups")
x11()
p <- ggplot(data = df.m, aes(x=variable, y=value)) +
  geom_boxplot(aes(fill=groups))
p + facet_wrap( ~ variable, scales="free")





detach(dataframesouriswork1)### on sort de la session, on se détache
rm(list=ls())### permet d'éliminer les éléments uploadés lors d'une analyse précédente= session précédente
ls()### verification que la liste est vide ou dan RStudio que la fenêtre "Envoronnement est vide#

