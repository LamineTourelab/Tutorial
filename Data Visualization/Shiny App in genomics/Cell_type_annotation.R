library(SingleR)
library(rstudioapi)
library(Seurat)
library(ggplot2)
library(tidyverse)
library(celldex)


# cell type annotation


# If first time running this script, the "celldex" package may ask you this
# question:

# /Users/(user name)/Library/Caches/org.R-project.R/R/ExperimentHub
#  does not exist, create directory? (yes/no): 

# type "yes" in the Console and continue.

# celldex also provides data sets for mouse tissues
# type celldex:: in the Rstudio console and hit "Tap" on the keyboard 
# to select other reference data sets

# example:
# mm_ref <- celldex::MouseRNAseqData()

# build function
func_get_annotation<- function(input_seurat)
  
{
  
  # download the reference data - The demo data was generated with human PBMCs.
  # Therefore, Human Primary Cell Atlas Data is suitable for this demonstration.
  hu_ref <- celldex::HumanPrimaryCellAtlasData()
  
  # Define the function
  DefaultAssay(input_seurat) <- "RNA"
  
  expr_matrix <- GetAssayData(input_seurat, 
                              slot = "data", 
                              assay = "RNA")
  
  cluster_id <- input_seurat@meta.data$seurat_clusters
  
  # Optional: annotate cells by groups
  prediction_by_cluster <-SingleR::SingleR(test = expr_matrix,
                                           ref = hu_ref, # make sure reference data is correct
                                           labels = hu_ref$label.main, # make sure reference data is correct
                                           clusters = cluster_id)
  
  # Annotation cells individually  
  prediction_by_cell <- SingleR::SingleR(test = expr_matrix,
                                         ref = hu_ref,  # make sure reference data is correct
                                         labels = hu_ref$label.main) # make sure reference data is correct
  
  # Save SingleR results to the Seurat object
  input_seurat@misc[["SingleR_results"]] <- prediction_by_cell
  
  # Annotation results
  cell_labels <- prediction_by_cell$labels
  
  names(cell_labels) <- rownames(prediction_by_cell$labels)
  
  # add annotation information to Seurat object under meta.data
  input_seurat <- AddMetaData(input_seurat, 
                              metadata = cell_labels,
                              col.name = "cell_type")
  
  # make cell types with less than 10 cells as "unknown"
  temp <- table(input_seurat$cell_type)[table(input_seurat$cell_type) < 10] %>% 
    names()
  
  input_seurat$cell_type[input_seurat$cell_type %in% temp] <- "unknow"
  
  # return Seurat object as output
  return(input_seurat)
  
} # end of func_get_annotation function
