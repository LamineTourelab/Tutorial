#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################### SCRIPT MOTHUR ###############
# 05-12-2017
# Adapté au Miseq
#####################################################

### Definition des répertoires de travail ###
# ——> REMPLIR LES xxxxx
set.dir(input=/xxx/xxx/xxx/xxx)
set.dir(output=/xxx/xxx/xxx/xxx)
# Parfois ça ne fonctionne pas correctement...
# Placer le programme Mothur dans le meme repertoire que vos donnees

### Realiser les contigs à partir des fichiers R1 et R2 ###
# créer un fichier *.files comme suivant (sur TextEdit)
#	echA	A_2_14_S1_L001_R1_001.fastq	A_2_14_S1_L001_R2_001.fastq
#	echB	B_2_14_S1_L001_R1_001.fastq	B_2_14_S1_L001_R2_001.fastq
# etc. 
# où les noms des échantillons est associé aux deux séquences (R1 et R2) du Miseq
# ——> REMPLIR LES xxxxx
make.contigs(file=stability.files)
summary.seqs(fasta=current)

### Nettoyage des sequences ###
# Apres le summary.seqs precedent on voit que les sequences font presques toutes 250pb
# on coupe pour ne garder que les séquences inferieurs à 275 bp
# ce qui nous permet d'enlever des sequences anormalement longues
screen.seqs(fasta=current, group=current, maxambig=0, maxlength=275)
summary.seqs(fasta=current)

### Reduction du fichier fasta en ne gardant qu’une ligne de chaque sequence
# création d’un fichier « name »
unique.seqs()
summary.seqs()

### Generation d’un tableau contenant le nom des séquences uniques (en lignes) 
# et le nom des groupes ou echantillons (en colonnes)
# chaque case du tableau contient le nombre de fois ou la sequence a été vu pour chaque groupe
count.seqs(name=current, group=current)
summary.seqs(fasta=current, count=current)

### Alignement des séquences sur une base de reference ###
# La base de reference choisi est la base SILVA (fasta aligne) qui a été préalablement
# réduite en rapport avec les primers utilises
# pour ce faire faire d'abord un alignement avec la base complete et faire un summary.seqs pour identifier le "start" et le "end" de vos séquences
# puis lancer la commande suivante (remplir les xx grace au summary.seqs precedent
# pcr.seqs(fasta=silva.bacteria.fasta, start=xx, end=xx, keepdots=F)
# renommer la nouvelle base réduite 
# rename.file(input=silva.bacteria.pcr.fasta, new=silva.v4.fasta)
# vous pouvez vous passer de cette étape, elle permet simplement de gagner du temps de calcul
# pour aligner nos sequences
align.seqs(fasta=current, reference=silva.v4.fasta)
summary.seqs(fasta=current, count=current)

### Nettoyage des séquences qui n’ont pas ete alignes ###
screen.seqs(fasta=curent, count=current, summary=current, start=1968, end=11550, maxhomop=8)
summary.seqs(fasta=current, count=current)

### Suppression des colonnes contenant des gaps (-) et
# Reduction du fichier fasta en ne gardant qu’une ligne de chaque sequence ###
# création d’un fichier « name »
filter.seqs(fasta=current, vertical=T, trump=.)
unique.seqs(fasta=current, count=current)

### Denoising des sequences ###
# Toujours dans le but de gagner du temps de calcul, on regroupe les sequences qui ont une différence de 1% (1 pour 100bp)
# Cette différence est dû aux erreurs de la polymerase
# dans notre cas pour des sequences de 250 bp on arrondi a 2 differences
pre.cluster(fasta=current, count=current, diffs=2)

### Recherche de chimeres ###
chimera.uchime(fasta=current, accnos=current, count=current)

### Suppression des chimeres ###
remove.seqs(fasta=current, accnos=current, count=current)
summary.seqs(fasta=current, count=current)

### Classification des sequences ###
# On utilise le RDP Classifier et la base RDP la plus récente (mai 2015)
classify.seqs(fasta=current, count=current, reference=trainset9_032015.pds.fasta, taxonomy=trainset9_032015.pds.tax, cutoff=80)

### Suppression de séquences non bactériennes ###
remove.lineage(fasta=current, count=current, taxonomy=current, taxon=Chloroplast-Mitochondria-unknown-Archea-Eukaryota)

### Regroupement des séquences par OTU ###
# Attention ceci est l’étape la plus longue du script (plusieurs jours quand on travaille sur plusieur dizaines d'echantillons)
cluster.split(fasta=current, count=current, taxonomy=current, splitmethod=fasta, cluster=f, taxlevel=5, cutoff=0.15, large=T)
cluster.split(file=current)

### Creation de la matrice d’OTU a 3% ###
make.shared(list=current, count=current, label=0.03)

### Classification des OTU ###
classify.otu(list=current, count=current, taxonomy=current, label=0.03)

### il n'y pas d'accent sur ce script car la bio-informatique deteste les accents !!!!!!
### bon courage a tous
### n'hesitez pas a me contacter par mail: alexandre.dos-santos@inserm.fr

