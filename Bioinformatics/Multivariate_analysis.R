

##### Lamine TOURE M2 OMICs ##############
####### LE 04/01/2021 Analyse lipidomique ###############


setRepositories()### permet de d?finir le lieu o? sont entrepos?s les packages
install.packages("Amelia")

library("beeswarm")
library("reshape2")
library("ggplot2")
#library("party")
library("cluster")
library("class")
library("gplots")
#library("gtools")
library("RColorBrewer")
library("pvclust")
require("pvclust")
require("rpart")
library("FactoMineR")
require("FactoMineR")
library("multtest")
#library("randomForest")
#require("randomForest")
#library("varSelRF")
#require("varSelRF")
#library("ROCR")
#library("pROC")
#library("AUCRF")
#library("languageR")
#library("designGG")
library("party") 
#library("Hmisc")
library("heatmap3")
#library("pamr")
library(factoextra)
library(FactoMineR)
library(pheatmap)

rm(list=ls())### permet d'?liminer les ?l?ments upload?s lors d'une analyse pr?c?dente= session pr?c?dente
ls()### verification que la liste est vide ou dan RStudio que la fen?tre "Envoronnement est vide#



## Il y'a des packages que vous n'utiliserez peut-être pas mais vaut mieux les telecharger

################################## Charger les donnes ############

RawData_lipido <- read.csv2("~/M2 Biomics/TPi/Data_lipido/RawData_lipido.csv", row.names=1)

str(RawData_lipido) ### pour voir la structure des données

RawData_lipido$Annee=as.factor(RawData_lipido$Annee)

###### Creer une variable contenant que les lipides (quantitatif)
lipids<- RawData_lipido[,-1]

 ### le package Amelia permet de representer les donnees manquantes
library(Amelia)
missmap(lipids)

#### A premier vu il y'a beacoup de donnes manquantes. 
#### Pour comparer les annees on doit supprimer les NA sur les colonnes
### le probleme que j'ai est que la commande na.omit() ne supprime que les na en suivant les lignes 
### Donc je vais transposer le tableau pour palier a ce probleme sinon toutes mes colonnes seront supprimer
data=t(lipids)
data=na.omit(data)

## une fois les NA enleves il suffit de retransposer la variable pour avoir le bon tableau

data=t(data)
### Les donnees manquantes sont supprimees 
##################### Let's go pour l'exploration des donnees #####################
#### Charger les métadonnees #####

dataglyco <- read.csv("~/M2 Biomics/TPi/Data_lipido/dataglyco.csv", sep=";")

## Clustering
library(cluster)
x11()
par(mfrow=c(1,2))
dv <- diana(data, metric = "manathan", stand = TRUE)
plot(dv)

library(gplots)
x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(data, Colv=as.dendrogram(dv), col = greenred(256), scale="column", margins=c(5,10), density="density", xlab = "Lipides", ylab= "Souris", main = "heatmap(Lipidomic souris)",breaks=seq(-1.5,1.5,length.out=257))


Metadata <- read.csv("~/M2 Biomics/TPi/Data_lipido/Metadata.csv", sep=";")


RawData_lipido$NumS=row.names(RawData_lipido)
View(RawData_lipido)

RawData_lipido$NumS

RawData_metadata=merge(RawData_lipido,Metadata)
head(RawData_metadata)
library(pheatmap
        )


library(corrplot)
dij = as.matrix( dist( data ) )
x11()
corrplot( dij ,is.corr = FALSE)
x11()
dij = dist( data, method="euclidian" )
hc = hclust( dij, method="complete")
plot( hc )

arow = data.frame( Genotype = RawData_metadata$Genotype, RawData_metadata$Age_mois )
rownames(arow)=rownames(data)

pheatmap( data, annotation_row=arow, cluster_rows=F, cluster_cols = F )
pheatmap( data, annotation_row=arow, cluster_rows=T, cluster_cols = T )

#### Variance des donnees par rapport a la moyenne 

MaVar= apply(data, 2, function(x) {sd(x)/mean(x)})
hist(MaVar)

# Suppression des lipides avec une variance < 0,5
TrieVar= which(MaVar>0.5)
MaVar1=as.data.frame(TrieVar)
#### On va trier les donnees pour ne garder que les 71 qui varient le plus 
data1 = data[ , rownames(MaVar1) ]

dim(data1)
library(pheatmap)

#### On refait la heatmap en standardisant les donnees pour ne pas que les PC l'emportent
x11()
pheatmap( data1, annotation_row=arow, cluster_rows=F, cluster_cols = T, scale = 'column' )

############ Clustering ##############
dv <- diana(data1, metric = "manathan", stand = TRUE)
plot(dv)

x11()
###IMPORTANT= HEATMAP CONSTRUIT LE HCLUST AVEC TOUS LES LIPIDES SAUF TOTAUX ET LE p-VALUES HERARCHICAL CLUSTERING
heatmap.2(data1, Colv=as.dendrogram(dv), col = greenred(256), scale="column", margins=c(5,10), density="density", xlab = "Lipides", ylab= "Souris", main = "heatmap(Lipidomic souris)",breaks=seq(-1.5,1.5,length.out=257))

arow1 = data.frame( Genotype = RawData_metadata$Genotype, RawData_metadata$Age_mois,Annee=RawData_metadata$Annee )
rownames(arow1)=rownames(data1)

pheatmap( data1, annotation_row=arow1, cluster_rows=F, cluster_cols = T, scale = 'column' )


arow2 = data.frame(Annee=RawData_metadata$Annee )
rownames(arow2)=rownames(data1)

pheatmap( data1, annotation_row=arow2, cluster_rows=F, cluster_cols = T, scale = 'column' )


arow3 = data.frame(Annee=RawData_metadata$Annee )
rownames(arow3)=rownames(data1)

pheatmap( data[,1:8], annotation_row=arow3, cluster_rows=F, cluster_cols = T, scale = 'column' )


arow3 = data.frame(Annee=RawData_metadata$Annee )
rownames(arow3)=rownames(data1)

pheatmap( data[,4:71], annotation_row=arow3, cluster_rows=F, cluster_cols = T, scale = 'column' )



Annee= data.frame(Annee=Rawdata_metada$Annee )
rownames(Annee)=rownames(data1)

Geno= data.frame(Genotype=Rawdata_metada$Genotype )
rownames(Geno)=rownames(data1)

pheatmap( data1, annotation_row=Annee, cluster_rows=T, cluster_cols = T, scale = 'column' )

pheatmap( data1, annotation_row=Annee, cluster_rows=T, cluster_cols = F, scale = 'column' , cutree_rows = 5)

############ Principal Component Analysis ############
########### Vous n'avez peut-etre pas besoin de package "broom" mais 
# si vous n'arrivez pas representer l'ACP vous pouvez le telecharger #####
library(broom)
library(factoextra)

library(FactoMineR)

acp=PCA(data[,9:142], scale.unit=T, graph=F )
acp$eig

fviz_eig(acp)

fviz_pca_biplot(acp)
fviz_pca_ind(acp)
fviz_pca_biplot( acp, geom.ind = "text" )

fviz_pca_ind( acp, fill.ind = RawData_metadata$Annee,geom.ind = "point", 
                 pointshape=21,addEllipses = F,pointsize=4 )

fviz_pca_ind( acp, fill.ind = RawData_metadata$Genotype,geom.ind = "point", 
              pointshape=21,addEllipses = F,pointsize=4 )

fviz_pca_ind( acp, fill.ind = RawData_metadata$Age_mois,geom.ind = "point", 
              pointshape=21,addEllipses = F,pointsize=4 )

fviz_pca_biplot(acp, fill.ind = RawData_metadata$Annee,geom.ind = "point", 
                pointshape=21,addEllipses = F,pointsize=4)

dimdesc(acp)$Dim.1$quanti[,1] # permet de s?parer les MCD des autres
dimdesc(acp)$Dim.2$quanti[,1]

####### ACP sur les 71 lipides avec avec une bonne variance ##################

acp1=PCA(data1, scale.unit=T, graph=F )
acp1$eig
fviz_eig(acp1)
fviz_pca_ind(acp1)

fviz_pca_ind( acp1, fill.ind = RawData_metadata$Annee,geom.ind = "point", 
              pointshape=21,addEllipses = F,pointsize=4 )

fviz_pca_ind( acp1, fill.ind = RawData_metadata$Genotype,geom.ind = "point", 
              pointshape=21,addEllipses = F,pointsize=4 )

fviz_pca_ind( acp1, fill.ind = RawData_metadata$Age_mois,geom.ind = "point", 
              pointshape=21,addEllipses = F,pointsize=4 )

dimdesc(acp1)$Dim.1$quanti[,1] # permet de s?parer les MCD des autres
dimdesc(acp1)$Dim.2$quanti[,1]

#### Correlation ######"

library(corrplot)

M=cor(data1)
corrplot(M,type = "upper",order = "hclust",addCoefasPercent = TRUE)

diag(M) = 0
corrplot( M, tl.srt=45, tl.cex=1.6 )

abs( M > 0.9 )
ix = apply( abs( M > 0.9), 1, any )
ix
# regardez a quelle ligne cela correspond et si vous comprenez
# ce que R calcule
 M1 = M[ ix,ix] # on conserve la matrice initiale
corrplot( M1 ,type = "upper",order = "hclust",addCoefasPercent = TRUE)


dij = as.matrix( dist( data1 ) )
corrplot( dij, is.corr=F,type = "upper" ,order = "hclust",addCoefasPercent = TRUE)

dij = dist( data, method="euclidian" )
hc = hclust( dij, method="complete")
plot( hc )



################## LipidR Pour analyser les lipides #########

BiocManager::install("lipidr")

usedVar =cbind("PC","PE","PS","PI","LPE","LPC","PG","SM")
 #############   LDA Analysis ###########
### Linear discriminant analysis (LDA): Uses linear combinations of predictors 
#to predict the class of a given observation. Assumes that the predictor variables (p) 
#are normally distributed and the classes have identical variances 
#(for univariate analysis, p = 1) or identical covariance matrices 
#(for multivariate analysis, p > 1).

#### Site du script #####
#http://www.sthda.com/english/articles/36-classification-methods-essentials/146-discriminant-analysis-essentials-in-r/


####### LDA analysis y'a til effet annne #####
library(tidyverse)
library(caret)
theme_set(theme_classic())

data2=data
data2=as.data.frame(data2)
data2$Annee=RawData_metadata$Annee


set.seed(123)
training.samples <- data2$Annee %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- data2[training.samples, ]
test.data <- data2[-training.samples, ]

##Estimate preprocessing parameters
preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)


library(MASS)
# Fit the model
model <- lda(Annee~., data = train.transformed)
# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class==test.transformed$Species)

library(MASS)
model <- lda(Annee~., data = train.transformed)
model

plot(model)

predictions <- model %>% predict(test.transformed)
names(predictions)

# Predicted classes
head(predictions$class, 6)
# Predicted probabilities of class memebership.
head(predictions$posterior, 6) 
# Linear discriminants
head(predictions$x, 3)

library(ggplot2)
lda.data <- cbind(train.transformed, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = Annee))



####### LDA analysis : Y'a til effet age#####

library(tidyverse)
library(caret)
theme_set(theme_classic())

data3=data
data3=as.data.frame(data3)
data3$Age=RawData_metadata$Age_mois

set.seed(123)
training.samples <- data3$Age %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- data3[training.samples, ]
test.data <- data3[-training.samples, ]

##Estimate preprocessing parameters
preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)


library(MASS)
# Fit the model
model <- lda(Annee~., data = train.transformed)
# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class==test.transformed$Species)

library(MASS)
model <- lda(Age~., data = train.transformed)
model

plot(model)

predictions <- model %>% predict(test.transformed)
names(predictions)

# Predicted classes
head(predictions$class, 6)
# Predicted probabilities of class memebership.
head(predictions$posterior, 6) 
# Linear discriminants
head(predictions$x, 3)

library(ggplot2)
lda.data <- cbind(train.transformed, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = Age))


########## LDA Analysis : y'a t'il effet Genotype#######
library(tidyverse)
library(caret)
theme_set(theme_classic())

data4=data
data4=as.data.frame(data4)
data4$Genotype=RawData_metadata$Genotype


set.seed(123)

training.samples <- data4$Genotype %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- data4[training.samples, ]
test.data <- data4[-training.samples, ]

##Estimate preprocessing parameters
preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)


library(MASS)
# Fit the model
model <- lda(Genotype~., data = train.transformed)
# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class==test.transformed$Species)

library(MASS)
model <- lda(Genotype~., data = train.transformed)
model

plot(model)

predictions <- model %>% predict(test.transformed)
names(predictions)

# Predicted classes
head(predictions$class, 6)
# Predicted probabilities of class memebership.
head(predictions$posterior, 6) 
# Linear discriminants
head(predictions$x, 3)

library(ggplot2)
lda.data <- cbind(train.transformed, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = Genotype))


####### Volcano plot #######

avg.t2d <- apply(data4[,data4$Genotype==" WT"], 1, mean)
avg.healthy <- apply(data4[,data4$Genotype==" TAU"], 1, mean)
foldch <- avg.t2d - avg.healthy

Fc=as.data.frame(foldch)

logt2dp <- -log(t2dpvals, base=10)
plot(foldch, logt2dp, pch=1, xlab="Log2 Fold Change (L/S)",
     ylab="-Log10(P-val)", main="Volcano Plot", col = ifelse(adj_pvals<0.05, 'green', 'black'))


################Boxplot ggplot2##########
datab=as.data.frame(data)

bxp_year_PC <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$PC, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of PC's abundance per Year",x=" Years", y = "PC_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_PC

bxp_year_PE <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$PE, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of PE's abundance per Year",x=" Years", y = "PE_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_PE

bxp_year_PS <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$PS, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of PS's abundance per Year",x=" Years", y = "PS_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_PS

bxp_year_PI <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$PI, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of PI's abundance per Year",x=" Years", y = "PI_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_PI

bxp_year_LPE <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$LPE, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of LPE's abundance per Year",x=" Years", y = "LPE_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_LPE

bxp_year_LPC <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$LPC, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of LPC's abundance per Year",x=" Years", y = "LPC_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_LPC

bxp_year_PG <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$PG, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of PG's abundance per Year",x=" Years", y = "PG_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_PG

bxp_year_SM <- ggplot(datab, aes(x = RawData_metadata$Annee, y = RawData_metadata$SM, fill = RawData_metadata$Genotype)) +     geom_boxplot()+ labs(title="Boxplot of SM's abundance per Year",x=" Years", y = "SM_Abandunce", fill="Genotype") + scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
bxp_year_SM

library(gridExtra)
grid.arrange(bxp_year_PC, bxp_year_LPC, bxp_year_LPE, bxp_year_PE, bxp_year_PG, bxp_year_PI, bxp_year_PS, bxp_year_SM)
