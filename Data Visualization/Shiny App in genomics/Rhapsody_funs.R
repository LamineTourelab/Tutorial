library(Seurat)
library(patchwork) #The goal of patchwork is to make it ridiculously simple to combine separate ggplots into the same graphic.
library(S4Vectors)
library(tidyverse)
library(ggplot2)
library(DoubletFinder) 
library(SeuratWrappers)
library(tidyverse)

# ==============================================================================. Curate the Seurat objects
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


# ==============================================================================. Data normalisation, scaling and finding clusters

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

# ==============================================================================. cell type annotation

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

# ==============================================================================. Find doublets

# BD provided doublet rates with different cell load numbers
rhapsody_doublet_rate <- data.frame(
  "cell_num" = c(100,500,1000*(1:20)), 
  "rate" = c(0, 0.1, 0.2, 0.5, 0.7, 1, 
             1.2, 1.4, 1.7, 1.9, 2.1, 
             2.4, 2.6, 2.8, 3.1, 3.3, 
             3.5, 3.8, 4, 4.2, 4.5 , 4.7))

# Build a linear model to calculate theoretical doublet rate
model_rhap <- lm(rate ~ cell_num, 
                 rhapsody_doublet_rate)

# define function
func_get_doublets <- function(seuratObj,
                              est_doublet_model = model_rhap,
                              pc = 1:15) # number of PC components to be used
{
  
  DefaultAssay(seuratObj) <- "RNA"
  
  # Find pK values
  sweep.res.list <- paramSweep(seuratObj, 
                               PCs = pc, 
                               sct = F)
  
  sweep.stats <- summarizeSweep(sweep.res.list, 
                                GT = FALSE)
  
  bcmvn <- find.pK(sweep.stats)
  
  pK_bcmvn <- bcmvn$pK[which.max(bcmvn$BCmetric)] %>% 
    as.character() %>% 
    as.numeric()
  
  # estimate doublet rate based on cell number
  DoubletRate = predict(est_doublet_model, 
                        data.frame(cell_num = dim(seuratObj)[2]))/100
  
  nExp_poi <- round(DoubletRate*ncol(seuratObj)) 
  
  seuratObj <- doubletFinder(seuratObj, 
                             PCs = pc, 
                             pN = 0.25, 
                             pK = pK_bcmvn, 
                             nExp = nExp_poi, 
                             reuse.pANN = F, 
                             sct = F)
  
  temp1 <- grepl("DF.classifications", 
                 colnames(seuratObj@meta.data), 
                 ignore.case = T)
  
  colnames(seuratObj@meta.data)[temp1] <- "doublet_check"
  
  seuratObj$doublet_check <- seuratObj$doublet_check
  
  # return output
  return(seuratObj)
  
} # end of function

# ==============================================================================. Finding marker genes
# Build function
func_get_marker_genes <- function(input_seurat,
                                  p_adj_cutoff = 0.05,
                                  log2FC_cutoff = 1,
                                  view_top_X_genes = 5)
  
{
  # Find marker genes for each cluster group against the rest
  Seurat::Idents(input_seurat) <- "seurat_clusters"
  
  cluster_DGE <- SeuratWrappers::RunPrestoAll(input_seurat, 
                                              assay = "RNA", 
                                              only.pos = FALSE, 
                                              verbose = FALSE)
  
  # Find marker genes for each cell type against the rest
  Seurat::Idents(input_seurat) <- "cell_type"
  
  Cell_type_DGE <- SeuratWrappers::RunPrestoAll(input_seurat, 
                                                assay = "RNA", 
                                                only.pos = FALSE, 
                                                verbose = FALSE)
  
  # example of taking the top X genes in each DGE group and removing the duplicates
  cluster_DGE <- cluster_DGE[abs(cluster_DGE$avg_log2FC) > log2FC_cutoff & 
                               cluster_DGE$p_val_adj < p_adj_cutoff, ]
  
  Cell_type_DGE <- Cell_type_DGE[abs(Cell_type_DGE$avg_log2FC) > log2FC_cutoff & 
                                   Cell_type_DGE$p_val_adj < p_adj_cutoff, ]
  
  top_genes_cluster <- cluster_DGE %>% 
    group_by(cluster)%>% 
    slice_max(n = view_top_X_genes, 
              order_by = avg_log2FC) %>% 
    dplyr::pull(gene) %>% 
    unique()
  
  top_genes_type <- Cell_type_DGE %>% 
    group_by(cluster)%>% 
    slice_max(n = view_top_X_genes, 
              order_by = avg_log2FC) %>% 
    dplyr::pull(gene) %>% 
    unique()
  
  # return output as a list
  return(list(DGEs_cluster = cluster_DGE, 
              DGEs_cell = Cell_type_DGE, 
              top_cluster_gene = top_genes_cluster, 
              top_cell_gene = top_genes_type))
  
} # end of function

# ==============================================================================. Slingshot pseudotime analysis

# Define function
func_slingshot <- function(input_seurat)
{
  
  # group cells at early states as start_clus
  
  # This is just an example. Please define root cells that are biologically 
  # correct for your experiment.
  # Note: if ids is an empty object, then this function may generate error.
  # You can use this line instead:       
  
  #    demo_sce <- slingshot::slingshot(demo_sce, 
  #                                     clusterLabels = "cell_type",
  #                                     reducedDim = "PCA", 
  #                                     allow.breaks = F)
  
  # and remove the following four lines, "Seurat::Idents(input_seurat) <- ..",
  # "ids <- ..."
  # "input_seurat$cell_type <- .." and 
  # "input_seurat$cell_type[input_seurat$cell_type %in% ids] <- .."
  
  # for more information, please visit 
  # http://www.bioconductor.org/packages/release/bioc/vignettes/slingshot/inst/doc/vignette.html
  
  Seurat::Idents(input_seurat) <- "cell_type"
  
  ids <- unique(input_seurat$cell_type) %>% 
    .[grep(paste("CMP", 
                 "GMP", 
                 "HSC", 
                 "CD34", 
                 "stem", 
                 "iPS", 
                 sep = "|"), 
           ., 
           ignore.case = T)]
  
  input_seurat$cell_type[input_seurat$cell_type %in% ids] <- "start_clus"
  
  # convert Seurat to SingleCellExperiment class
  demo_sce <- Seurat::as.SingleCellExperiment(input_seurat)
  
  # Perform slingshot analysis
  temp <- tryCatch(
    {
      
      # PCA, tSNE and UMAP are all accepted for slingshot analysis. 
      # However, PCA has more dimensions. Hence, it's recommended by Slingshot Author.
      
      # Subset first 6 PC components
      reducedDim(demo_sce, 
                 type = "PCA", 
                 WithDimnames = TRUE) <- reducedDim(demo_sce,
                                                    type = "PCA")[, 1:6]
      
      demo_sce <- slingshot::slingshot(demo_sce, 
                                       clusterLabels = "cell_type",
                                       start.clus = "start_clus",
                                       reducedDim = "PCA", 
                                       allow.breaks = F)
    },
    error = function(e){
      
      # if there is an issue with PC selection, switch reduction method to UMAP
      demo_sce <- slingshot::slingshot(demo_sce, 
                                       clusterLabels = "cell_type",
                                       start.clus = "start_clus",
                                       reducedDim = "UMAP", 
                                       allow.breaks = F)
      
    }
  )
  
  # return SCE object
  return(temp)
  
} # end of function