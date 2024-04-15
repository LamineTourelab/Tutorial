library(Seurat)
library(patchwork) #The goal of patchwork is to make it ridiculously simple to combine separate ggplots into the same graphic.
library(S4Vectors)
library(tidyverse)
library(ggplot2)


# Curate the Seurat objects
func_get_AbSeq <- function(demo_seurat){
  AbSeq_list <- rownames(demo_seurat)[grepl("pAbO", rownames(demo_seurat))]
  
  # get RNA gene list
  RNA_list <- rownames(demo_seurat)[!grepl("pAbO", rownames(demo_seurat))]
  
  # subset demo_seurat and demo_seurat by AbSeq and RNA gene lists
  demo_AbSeq <- subset(demo_seurat,features=AbSeq_list)
  
  demo_seurat <- subset(demo_seurat,features=RNA_list)
  # create new Seurat objects
  demo_seurat@assays[["AB"]] <- GetAssay(demo_AbSeq,assay = "RNA")
  
  # change default assay to AB
  DefaultAssay(demo_seurat) <- "AB"
  demo_seurat$nCount_AB <- colSums(x = demo_seurat, slot = "counts")  # nCount_RNA
  demo_seurat$nFeature_AB <-  colSums(x = GetAssayData(object = demo_seurat, slot = "counts") > 0)
  
  # remove demo_AbSeq
  remove(demo_AbSeq)
  
  return(demo_seurat)
} # end of func_get_AbSeq function


# Data normalisation, scaling and finding clusters

func_quick_process <- function(demo_seurat,
                               ab_pc_num = 10, # number of PCA components to use for protein
                               rna_pc_num = 15, # number of PCA components to use for RNA
                               ab_reduction_res = 0.8, # cluster resolution for protein
                               rna_reduction_res = 0.8) # cluster resolution for RNA
  
{
  
  # check if the Seurat object has protein assay
  if("AB" %in% Seurat::Assays(demo_seurat))
    
  {  
    Seurat::DefaultAssay(demo_seurat) <- 'AB'
    
    # Normalize and scale data
    demo_seurat <- demo_seurat %>% 
      Seurat::NormalizeData()  %>% 
      Seurat::FindVariableFeatures() %>% 
      Seurat::ScaleData() 
    
    # perform PCA
    demo_seurat <- Seurat::RunPCA(object = demo_seurat, 
                                  reduction.name = 'apca')
    
    # perform UMAP
    demo_seurat <- Seurat::RunUMAP(demo_seurat, 
                                   reduction = 'apca', 
                                   dims = 1:ab_pc_num, 
                                   assay = 'AB', 
                                   reduction.name = 'adt.umap',
                                   reduction.key = 'adtUMAP_')
    
    # Find clusters
    demo_seurat <- Seurat::FindNeighbors(demo_seurat, 
                                         reduction = "apca", 
                                         dims = 1:ab_pc_num)
    
    demo_seurat <- Seurat::FindClusters(demo_seurat, 
                                        resolution = ab_reduction_res, 
                                        graph.name = "AB_snn")
    
  } # end of if statement
  
  # Change default assay to RNA
  Seurat::DefaultAssay(demo_seurat) <- "RNA"
  
  # Calculate percentages of mitochondrial gene expression for every cell. If it is a
  #  targeted sequencing, then the output of this commend line is all zeros.
  demo_seurat <- Seurat::PercentageFeatureSet(demo_seurat, 
                                              pattern = "^MT[-|.]", 
                                              col.name = "percent.mt") 
  
  # find top most variant genes
  demo_seurat <- demo_seurat %>% 
    Seurat::NormalizeData() %>% 
    Seurat::FindVariableFeatures(., 
                                 selection.method = "vst")
  
  # scale data
  demo_seurat <- Seurat::ScaleData(demo_seurat, 
                                   verbose = FALSE)
  
  # perform PCA
  demo_seurat <- Seurat::RunPCA(demo_seurat, 
                                npcs = rna_pc_num, 
                                verbose = FALSE)
  
  # perform UMAP
  demo_seurat <- Seurat::RunUMAP(demo_seurat, 
                                 reduction = "pca", 
                                 dims = 1:rna_pc_num)
  
  # Find clusters
  demo_seurat <- Seurat::FindNeighbors(demo_seurat, 
                                       reduction = "pca", 
                                       dims = 1:rna_pc_num)
  
  demo_seurat <- Seurat::FindClusters(demo_seurat, 
                                      resolution = rna_reduction_res)
  
  demo_seurat <- Seurat::BuildClusterTree(demo_seurat)
  
  # return Seurat object as output
  return(demo_seurat)
  
} # end of func_quick_process function
