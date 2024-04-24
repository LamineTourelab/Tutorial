library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)
library(ggrepel)
library(plotly)
library(gridExtra)
library(tidyr)
library(tidyverse)
library(DT)
library(dplyr)
library(factoextra) # for pca visualization 
library(ggpubr)
library(NbClust) # Kmeans best cluster number
library(FactoMineR)  # PCA
library(enrichR) #Enrichment
library(gplots) # Use heatmap.2
library(corrplot)
library(mixOmics) # for the breast cancer dataset
library(Amelia) # for missing values visualization
library(igvShiny)
library(GenomicAlignments)
library(rtracklayer)
library(rstudioapi)
library(Seurat)
library(patchwork) #The goal of patchwork is to make it ridiculously simple to combine separate ggplots into the same graphic.
library(S4Vectors)
library(celldex) #BiocManager::install('celldex')
library(SingleR) #BiocManager::install('SingleR')
library(harmony)
library(DoubletFinder) #remotes::install_github('chris-mcginnis-ucsf/DoubletFinder')
library(SeuratWrappers) #remotes::install_github('satijalab/seurat-wrappers')
library(slingshot) #BiocManager::install('slingshot')
library(colorRamps)
library(CellChat) #remotes::install_github('sqjin/CellChat')
source("Rhapsody_funs.R")
source("Util.R")

## ==================================================================== Datasets ============================================================================================##
data(breast.TCGA) # from the mixomics package.
mRna = data.frame(breast.TCGA$data.train$mrna)
mRna$subtype = breast.TCGA$data.train$subtype
Transcriptomics_data <- readr::read_csv("https://raw.githubusercontent.com/LamineTourelab/Tutorial/main/Data%20Visualization/Shiny%20App%20in%20genomics/Data/Transcriptomics%20data.csv")
stock.genomes <- sort(get_css_genomes())
dbs <- listEnrichrDbs()
# ==================================================================== Options ===================================================================================================#
options(shiny.maxRequestSize = 50*1024^2) 
# ======================================================================  Ui. =================================================================================================##


dashHeader = dashboardHeader(title ="Shiny Bio for genomics data",
                             tags$li(a(href = 'https://github.com/LamineTourelab',
                                       icon("github"),
                                       title = "Autor Github"),
                                     class = "dropdown"),
                             tags$li(a(href = 'https://www.linkedin.com/in/lamine-toure',
                                       icon("linkedin"),
                                       title = "Autor linkedin"),
                                     class = "dropdown"),
                             tags$li(a(href = 'https://laminetourelab.github.io/',
                                       icon("blog"),
                                       title = "Autor Website"),
                                     class = "dropdown"),
                             tags$li(a(href = 'https://github.com/LamineTourelab/Tutorial/blob/main/Data%20Visualization/Shiny%20App%20in%20genomics/app.R',
                                       icon("code"),
                                       title = "Source Code"),
                                     class = "dropdown"),
                             tags$li(a(href = 'http://shinyapps.company.com',
                                       icon("power-off"),
                                       title = "Back to Apps Home"),
                                     class = "dropdown"),
                             tags$li(a(href = 'https://www.institut-necker-enfants-malades.fr/',
                                       img(src = 'inem.jpeg',
                                           title = "Company Home", height = "30px"),
                                       style = "padding-top:10px; padding-bottom:10px;"),
                                     class = "dropdown")
                             )
dashsidebar = dashboardSidebar(
  sidebarUserPanel("Lamine TOURE",
                   subtitle = a(href = "#", icon("circle", class = "text-success"), "Online"),
                   # Image file should be in www/ subdir
                   image = "logo.jpeg"
  ),
  sidebarMenu(
    menuItem(
      text = 'Home', 
      tabName = 'hometab',
      icon = icon('home', style = "color:#E87722")),
    
    menuItem(
      text = 'Graphs',
      tabName = 'Graphstab',
      icon = icon('chart-column', style = "color:#E87722")),
    
    menuItem(
      text = 'Statistics',
      tabName = 'Statistics',
      icon = icon('chart-line', style = "color:#E87722"),
      menuSubItem(tabName = 'acp', text = 'PCA'),
      menuSubItem(tabName = 'clustering', text = 'Clustering'),
      menuSubItem(tabName = 'statstest', text = "Statistical test")
    ),
    
    menuItem(
      text = 'DEA',
      tabName = 'diffexp',
      icon = icon('dna', style = "color:#E87722")),
    
    menuItem(
      text = 'Enrichment',
      tabName = 'enrich',
      icon = icon('dna', style = "color:#E87722")),
    
    menuItem(
      text = 'IGV',
      tabName = 'igv',
      icon = icon('dna', style = "color:#E87722")),
    
    menuItem(
      text = 'scRNAseq',
      tabName = 'scrnaseq',
      icon = icon('dna', style = "color:#E87722"),
      menuSubItem(tabName = '10X', text = '10x Genomic'),
      menuSubItem(tabName = 'rhapsody', text = 'BD Rhapsody')
      )
    
    
  ) #sidebarMenu
)#dashboardSidebar

dashbody <- dashboardBody(
  shinyjs::useShinyjs(),
  # =================================================================================================  Home
  tabItems(
    tabItem(tabName = 'hometab',
            h1('Shiny Bio for genomics data!'),
           # img(src = "inem.jpeg", height = 72, width = 72),
            p(style="text-align: justify;", strong('shinyBio:'), 'a shiny web app to easily perform popular visualization analysis for omics data. This is under development
              and you can see the new releases on this repository', a("LamineTourelab/Tutorial",href = "https://github.com/LamineTourelab/Tutorial"), br(), "See some example of outputs below:" 
              ),
           h3("Graphical tool for gene visualization"),
           br(), img(src = "Histogram_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "Box_plot_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = " Violin_plot_2024-03-27.png", align = "center", width = "500", height = "339"),
           img(src = "Linear_plot_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "Density_plot_2024-03-26.png", align = "center", width = "500", height = "339"),
           h3("PCA Analysis for clustering"),
           br(), img(src = "PCA_Component_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "PCA_Individuals_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "PCA_Importance_genes_2024-03-26.png", align = "center", width = "500", height = "339"),
           h3("Kmeans clustring analysis"),
           br(), img(src = "Kmeans_clusters_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "Kmeans_clustrs_annotated_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "Kmeans_Elbow_method_2024-03-26.png", align = "center", width = "500", height = "339"),
           h3("Differential expression analysis"),
           br(), img(src = "table_diff.png", align = "center", width = "500", height = "339"),
           img(src = "Volcanoplot_2024-03-26.png", align = "center", width = "500", height = "339"),
           h3("Graphical tool for gene enrichment analysis with Enrichr"),
           br(), img(src = "Enrichment_Analysis_2024-03-26.png", align = "center", width = "500", height = "339"),
           img(src = "Enrichrdatabase.png"),
           img(src = "Enrichment.png"),
           h3("Integrative Genome Visualization"),
           br(), img(src = "igv.png"),
    ),
    # ============================================================================================= Graph
    tabItem(tabName = 'Graphstab', 
            
            fluidPage(
              sidebarLayout(
                sidebarPanel(width = 3,
                             collapsible = TRUE,
                             title = 'Side bar',
                             status = 'primary', solidHeader = TRUE,
                             p(style="text-align: justify;",
                               "Here you can upload you own data by changing the mode test-data to own.", br(), "Maximum size = 50MB"),
                             selectInput("datasetgraph", "Choose a dataset:", choices = c("test-data", "own")),
                             fileInput(inputId = 'filegraph', 'Please upload a matrix file'),
                             p(style="text-align: justify;",
                               "Here you can choose a variable to plot and for coloring. By default you can select the cancer subtype (the last variable of the color list) for color variable."),
                             hr(style="border-color: blue;"),
                             # Placeholder for input selection
                             h4(strong("Histogram and boxplot panel")),
                             fluidRow(
                               column(4, selectInput(inputId ='Vartoplot', label = 'Waiting for data', choices = NULL)),
                               column(4, selectInput(inputId='VarColor',label = 'Waiting for data Color', choices = NULL))
                             ),
                             # Choose number of bins
                             sliderInput(inputId='histbins',
                                         label = 'please select a number of bins',
                                         min = 5, max = 50, value = 30),
                             hr(style="border-color: blue;"),
                             h4(strong("Linear & density plot panel")),
                             fluidRow(
                               column(4, selectInput(inputId='VarColor1',label = 'Waiting for data Color', choices = NULL)),
                               column(4, selectInput(inputId ='Vartoplot1', label = 'Waiting for data', choices = NULL)),
                               column(4, selectInput(inputId ='Vartofill', label = 'Waiting for data', choices = NULL))
                             )
                ),
                mainPanel(width = 9,
                          tabsetPanel(
                            tabPanel(title='Histogram ',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='Histplot',height = "600px"),
                                     h4(strong("Exporting the Histogram")),
                                     fluidRow(
                                       column(3,numericInput("width_png_hist","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_hist","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_hist","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_hist','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Box  plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='boxplot',height = "600px"),
                                     h4(strong("Exporting the Box plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_boxplot","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_boxplot","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_boxplot","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_boxplot','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Violin  plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='violinplot',height = "600px"),
                                     h4(strong("Exporting the Violin plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_violin","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_violin","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_violin","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_violin','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Linear plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='linearplot',height = "600px"),
                                     h4(strong("Exporting the linear plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_linearplot","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_linearplot","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_linearplot","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_linearplot','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Density plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='density',height = "600px"),
                                     h4(strong("Exporting the Density plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_densityplot","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_densityplot","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_densityplot","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_densityplot','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Data Table',
                                     DT::dataTableOutput(outputId = 'thetable',height = "600px"),
                                     # verbatimTextOutput("summarythetable")
                            )
                          ) #tabsetPanel
                ) #mainPanel
              ) #sidebarLayout
            ) # fluidPage
    ), #tabItem for graphs
    
    # # ================================================================================  Statistical Analysis.
    tabItem(tabName = 'acp',
            fluidPage(
              sidebarLayout(
                sidebarPanel(width = 3,
                             collapsible = TRUE,
                             title = 'Side bar',
                             status = 'primary', solidHeader = TRUE,
                             p(style="text-align: justify;",
                               "Here you can upload you own data by changing the mode test-data to own.
                    The should have as rownames the first column and the same rownames and dimension as the metadata file.", 
                               br(), "Maximum size = 50MB"),
                             selectInput("datasetstats", "Choose a dataset:", choices = c("test-data", "own")),
                             p(style="text-align: justify;","The uploading data should be a matrix without any factor column"),
                             fileInput(inputId = 'filestats', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             
                             p(style="text-align: justify;","Here you can upload you own metadata by changing the mode test-metadata to own-metadata.
                    The should have as rownames the first column and the same rownames and dimension as the dataset file above.", br(), "Maximum size = 50MB"),
                             selectInput("datasetstatsmetd", "Choose a meta-dataset:", choices = c("test-metadata", "own-metadata")),
                             p(style="text-align: justify;","The uploading data should be a file with factor column as metadata file."),
                             fileInput(inputId = 'filestatsmetd', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             # Placeholder for input selection
                             selectInput(inputId='Vartoplotstats',label = 'Waiting for metadata', choices = NULL ),
                             p(style="text-align: justify;","It  may take a little time for big dataset. Take a coffee!")
                             
                ), # sidebarPanel
                mainPanel(width = 9,
                          tabsetPanel(
                            tabPanel(title='PCA Component  plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='pcacomp',height = "600px"),
                                     h4(strong("Exporting the Scree plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_scree","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_scree","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_scree","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_scree','Download PNG'))
                                     )
                            ),
                            tabPanel(title='Individuals  plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='pcaind',height = "600px"),
                                     h4(strong("Exporting the Individuals PCA plot")),
                                     fluidRow(
                                       
                                       column(3,numericInput("width_pcaind", "Width of PDF", value=10)),
                                       column(3,numericInput("height_pcaind", "Height of PDF", value=8)),
                                       column(3),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPDF_pcaind','Download PDF'))
                                     ),
                                     
                                     fluidRow(
                                       column(3,numericInput("width_png_pcaind","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_pcaind","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_pcaind","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_pcaind','Download PNG'))
                                     )
                            ),
                            
                            tabPanel(title='PCA-biplot  plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='pcabiplot',height = "600px"),
                                     h4(strong("Exporting the PCA-biplot plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_biplot","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_biplot","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_biplot","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_biplot','Download PNG'))
                                     )
                            ),
                            
                            tabPanel(title='Gene Component association plot',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='variableimp',height = "600px"),
                                     h4(strong("Exporting the Gene Component association plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_genepca","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_genepca","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_genepca","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_genepca','Download PNG'))
                                     )
                            ),
                            
                            tabPanel(title='Data Table',
                                     DT::dataTableOutput(outputId = 'thetablestats')
                                     
                            ) 
                            
                          ) # tabsetPanel
                ) # mainPanel
              ) # sidebarLayout
            ) # fluidPage
            
    ), # tabItem for PCA
    # ========= clustring
    tabItem(tabName = 'clustering',
            fluidPage(
              sidebarLayout(
                sidebarPanel(width = 3,
                             collapsible = TRUE,
                             title = 'Side bar',
                             status = 'primary', solidHeader = TRUE,
                             p(style="text-align: justify;",
                               "Here you can upload you own data by changing the mode test-data to own.
                    The should have as rownames the first column and the same rownames and dimension as the metadata file.", 
                               br(), "Maximum size = 50MB"),
                             selectInput("datasetstatsclust", "Choose a dataset:", choices = c("test-data", "own")),
                             p(style="text-align: justify;","The uploading data should be a matrix without any factor column"),
                             fileInput(inputId = 'filestatsclust', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             
                             p(style="text-align: justify;","Here you can upload you own metadata by changing the mode test-metadata to own-metadata.
                    The should have as rownames the first column and the same rownames and dimension as the dataset file above.", br(), "Maximum size = 50MB"),
                             selectInput("datasetstatsmetdclust", "Choose a meta-dataset:", choices = c("test-metadata", "own-metadata")),
                             p(style="text-align: justify;","The uploading data should be a file with factor column as metadata file."),
                             fileInput(inputId = 'filestatsmetdclust', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             # Placeholder for input selection
                             selectInput(inputId='Vartoplotstatsclust',label = 'Waiting for metadata', choices = NULL ),
                             p(style="text-align: justify;","It  may take a little time for big dataset. Take a coffee!"),
                             hr(style="border-color: blue;"),
                             # Placeholder for input selection
                             h4(strong("Select the optimal number of cluster")),
                             # Choose number of bins
                             sliderInput(inputId='kmeansbins',
                                         label = 'Please select a number of clusters',
                                         min = 1, max = 20, value = 4),
                             
                ), # sidebarPanel
                mainPanel(width = 9,
                          tabsetPanel(
                            tabPanel(title='Hierarchical Clustering',
                                     #Placeholder for plot
                                     plotOutput(outputId = 'missmap',height = "600px"),
                                     h4(strong("Exporting the Missing values plot")),
                                     fluidRow(
                                       column(3,numericInput("width_png_missmap","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png_missmap","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG_missmap","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_missmap','Download PNG'))
                                     )
                            ),
                            tabPanel( title='Heatmap plot',
                                      #Placeholder for plot
                                      plotOutput(outputId='heatmap',height = "600px"),
                                      h4(strong("Exporting the Heatmap plot")),
                                      fluidRow(
                                        column(3,numericInput("width_png_heatmap","Width of PNG", value = 1600)),
                                        column(3,numericInput("height_png_heatmap","Height of PNG", value = 1200)),
                                        column(3,numericInput("resolution_PNG_heatmap","Resolution of PNG", value = 144)),
                                        column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_heatmap','Download PNG'))
                                      )
                            ),
                            tabPanel(title='K means',
                                     #Placeholder for plot
                                     fluidPage(
                                       plotlyOutput(outputId='kmeanscluster',height = "600px"),
                                       h4(strong("Exporting the Kmeans cluster plot")),
                                       fluidRow(
                                         column(3,numericInput("width_png_kmeanscluster","Width of PNG", value = 1600)),
                                         column(3,numericInput("height_png_kmeanscluster","Height of PNG", value = 1200)),
                                         column(3,numericInput("resolution_PNG_kmeanscluster","Resolution of PNG", value = 144)),
                                         column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_kmeanscluster','Download PNG'))
                                       )
                                     ),
                                     fluidPage(
                                       plotlyOutput(outputId='kmeansclusterannot',height = "600px"),
                                       h4(strong("Exporting the Kmeans annotated plot")),
                                       fluidRow(
                                         column(3,numericInput("width_png_kmeansclusterannot","Width of PNG", value = 1600)),
                                         column(3,numericInput("height_png_kmeansclusterannot","Height of PNG", value = 1200)),
                                         column(3,numericInput("resolution_PNG_kmeansclusterannot","Resolution of PNG", value = 144)),
                                         column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_kmeansclusterannot','Download PNG'))
                                       )
                                     ),
                                     fluidPage(
                                       plotlyOutput(outputId='kmeanselbow',height = "600px"),
                                       h4(strong("Exporting the Kmeans Elbow plot")),
                                       fluidRow(
                                         column(3,numericInput("width_png_kmeanselbow","Width of PNG", value = 1600)),
                                         column(3,numericInput("height_png_kmeanselbow","Height of PNG", value = 1200)),
                                         column(3,numericInput("resolution_PNG_kmeanselbow","Resolution of PNG", value = 144)),
                                         column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_kmeanselbow','Download PNG'))
                                       )
                                     ),
                                     
                            ),
                            tabPanel(title='Data Table',
                                     DT::dataTableOutput(outputId = 'thetablestatsclust')
                                     
                            ) 
                            
                          ) # tabsetPanel
                ) # mainPanel
              ) # sidebarLayout
            ) # fluidPage
            
    ), # tabItem for Clustering
    # ========= Statistical Test
    tabItem(tabName = 'statstest',
            fluidPage(
              sidebarLayout(
                sidebarPanel(width = 3,
                             collapsible = TRUE,
                             title = 'Side bar',
                             status = 'primary', solidHeader = TRUE,
                             p(style="text-align: justify;",
                               "Here you can upload you own data by changing the mode test-data to own.
                    The should have as rownames the first column and the same rownames and dimension as the metadata file.", 
                               br(), "Maximum size = 50MB"),
                             selectInput("datasetstatstest", "Choose a dataset:", choices = c("test-data", "own")),
                             p(style="text-align: justify;","The uploading data should be a matrix without any factor column"),
                             fileInput(inputId = 'filestatstest', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             
                             p(style="text-align: justify;","Here you can upload you own metadata by changing the mode test-metadata to own-metadata.
                    The should have as rownames the first column and the same rownames and dimension as the dataset file above.", br(), "Maximum size = 50MB"),
                             selectInput("datasetstatsmetdtest", "Choose a meta-dataset:", choices = c("test-metadata", "own-metadata")),
                             p(style="text-align: justify;","The uploading data should be a file with factor column as metadata file."),
                             fileInput(inputId = 'filestatsmetdtest', 'Please upload a matrix file',
                                       accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                             # Placeholder for input selection
                             selectInput(inputId='Vartoplotstatstest',label = 'Waiting for metadata', choices = NULL ),
                             p(style="text-align: justify;","It  may take a little time for big dataset. Take a coffee!")
                             
                ), # sidebarPanel
                mainPanel(width = 9,
                          tabsetPanel(
                            tabPanel(title='Anova',
                                     #Placeholder for plot
                                     plotlyOutput(outputId='stattest',height = "600px"),
                            ),
                            tabPanel(title='Data Table',
                                     DT::dataTableOutput(outputId = 'thetablestatstest')
                                     
                            ) 
                            
                          ) # tabsetPanel
                ) # mainPanel
              ) # sidebarLayout
            ) # fluidPage
            
    ), # tabItem for ststs  test
    # # ================================================================================  Differential expression Analysis.
    tabItem(tabName = 'diffexp',
            fluidPage(
              sidebarLayout(
              sidebarPanel(width = 2, height = 1170,
                           collapsible = TRUE,
                           title = 'Side bar',
                           status = 'primary', solidHeader = TRUE,
                           p(style="text-align: justify;",
                             "Here you can upload you own data by changing the mode test-data to own.", br(), "Maximum size = 50MB"),
                           selectInput("datasetdiff", "Choose a dataset:", choices = c("test-data", "own")),
                           p(style="text-align: justify;","The uploading data should be in the format :ID, logFC, Pvalue."),
                           fileInput(inputId = 'filediff', 'ID, logFC, Pvalue',
                                     accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                           hr(style="border-color: blue;"),
                           selectInput("diffenrich", "Choose a database:", choices = dbs$libraryName, selected = 'GO_Biological_Process_2023'),
                           actionButton('diffsubmit', strong('Submit Enrichment'))
                           # Placeholder for input selection
                           # fluidRow(
                           #   column(6, selectInput(inputId='Vartoplotdiff',label = 'Waiting for data plot',choices = NULL )),
                           #   column(6, checkboxGroupInput(inputId='Vardatabasediff',label = 'Choose database',choices = NULL ))
                           # )
              ),
              mainPanel( width = 10,
                         tabsetPanel(
                      tabPanel(title='Volcano plot',
                               p(),
                               textOutput("number_of_points"),
                               p(),
                               plotlyOutput(outputId = 'Volcanoplot',height = "600px"),
                               h4(strong("Exporting the Vocano plot")),
                               fluidRow(
                                 column(3,numericInput("width_png_volcano","Width of PNG", value = 1600)),
                                 column(3,numericInput("height_png_volcano","Height of PNG", value = 1200)),
                                 column(3,numericInput("resolution_PNG_volcano","Resolution of PNG", value = 144)),
                                 column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_volcano','Download PNG'))
                               )
                      ),
                      tabPanel(title='Data Table with links',
                               #Placeholder for plot
                               #selectInput("species", label = "Select relevant species",species,selected = "HUMAN"),
                               h2("Data table with database links:"),
                               div(DT::dataTableOutput(outputId = 'summarytable')),
                               
                      ),
                      tabPanel(title='Summary count table',
                               h2("The summary count table:"),
                               DT::dataTableOutput(outputId = 'summarytablecount')
                      ),
                      tabPanel(title='Gene Set Enrichment',
                               plotlyOutput(outputId = 'diffenrichplot',height = "600px")
                      )
              ) #tabsetPanel
              ) #mainPanel
              ) #sidebarLayout
            ) # fluidPage
            
    ), # tabItem for DEA
    # ================================================================================  IGV
    tabItem(
      tabName ='igv',
      sidebarPanel(width = 2,
                   selectInput("genomeChooser", "Choose a igv genome:", stock.genomes, selected = "hg38")),
      shinyUI(fluidPage(igvShinyOutput('igvShiny',height = "600px"), width = 10))
    ),
    # ================================================================================  Enrichment
    tabItem(
      tabName ='enrich',
      fluidPage(
        sidebarLayout(
          sidebarPanel(
            textAreaInput("genes", "Enter genes names (separed by a ',') :", placeholder = c('Enter a list of genes in this format : Gene1, Gene2, Gene3'), 
                          value = c('SFRP1, RELN, FLT1, GPC3, APOBEC3B, CD47, NTRK2, TLR8,FGF10, E2F1, HBEGF, SLC19A3, DUSP6, FOS, GNG5')),
            actionButton("submit", "Submit"),
            p(style="text-align: justify;",
              "Here you can choose a database to see the results in data table format and/or plot."),
            hr(style="border-color: blue;"),
            selectInput("databaseenrich", "Choose a database:", choices = dbs$libraryName, selected = 'GO_Biological_Process_2023')
          ),
          mainPanel(tabsetPanel(
            tabPanel(title = 'Enrichment',
                     #  verbatimTextOutput("results"),
                     DT::dataTableOutput(outputId = 'thetableenrich'),
            ),
            tabPanel(title = 'Plot',
                     plotlyOutput(outputId='gsea',height = "600px"),
                     h4(strong("Exporting the Enrichment plot")),
                     fluidRow(
                       column(3,numericInput("width_png_enrichr","Width of PNG", value = 1600)),
                       column(3,numericInput("height_png_enrichr","Height of PNG", value = 1200)),
                       column(3,numericInput("resolution_PNG_enrichr","Resolution of PNG", value = 144)),
                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_enrichr','Download PNG'))
                     )
            ),
            tabPanel(title = 'Enrichr database',
                     DT::dataTableOutput(outputId = 'enrichrdatabase'),
            )
          )
          ) # mainPanel
          
        )
      ) # fluidPage
      
    ),#tabItem
    # ================================================================================  10x Genomic
    tabItem(
      tabName = '10X',
      fluidPage(
        sidebarLayout(
          sidebarPanel(width = 3, 
                       p(style="text-align: justify;",
                         "Here you can upload you own data by changing the mode test-data to own.", br(), "Maximum size = 50MB"),
                       selectInput("dataset10x", "Choose a dataset:", choices = c("test-data", "own")),
                       p(style="text-align: justify;","The uploading data should be the .RDS output results of Seven Bridges platform."),
                       fileInput(inputId = 'file10x', 'Seurat RDS file from SevenBridges',
                                 accept=c('rds', '.rds')),
                       
          ),
          mainPanel( width = 9,
                     tabsetPanel(
                       tabPanel(title = 'Preprocessing ',
                                plotlyOutput(outputId='10xpreprocessing',height = "600px"),
                       ),
                       tabPanel(title = 'Cell annotation',
                                plotlyOutput(outputId='10xcellannotation',height = "600px"),
                       ),
                       tabPanel(title = 'Merge and Remove batch effect',
                                plotlyOutput(outputId='10xbatcheffect',height = "600px"),
                       ),
                       tabPanel(title = 'Finding doublets',
                                plotlyOutput(outputId='10xdoublet',height = "600px"),
                       ),
                       tabPanel(title = 'Finding marker genes',
                                plotlyOutput(outputId='10xmarkergenes',height = "600px"),
                       ),
                       navbarMenu(title = 'Further Analysis',
                                  tabPanel(title = 'Pseudotime Analysis',
                                           plotlyOutput(outputId='10xpseudotime',height = "600px"),
                                  ),
                                  tabPanel(title = 'Cell Communication',
                                           plotlyOutput(outputId='10xcellcommunication',height = "600px"),
                                  )
                       ) # navbarMenu
                     ) #tabsetPanel 
          ) #mainPanel
        ) # sidebarLayout
      ) # fluidPage
    ),
    # ================================================================================  BD Rhapsody
    tabItem(
      tabName = 'rhapsody',
      fluidPage(
        sidebarLayout(
          sidebarPanel(width = 3, 
                       p(style="text-align: justify;",
                         "Here you can upload you own data by changing the mode test-data to own.", br(), "Maximum size = 50MB"),
                       selectInput("datasetrhapsody", "Choose a dataset:", choices = c("test-data", "own")),
                       p(style="text-align: justify;","The uploading data should be the .RDS output results of Seven Bridges platform."),
                       fileInput(inputId = 'filerhapsody', 'Seurat RDS file from SevenBridges',
                                 accept=c('rds', '.rds')),
                       hr(style="border-color: blue;"),
                       # Choose number of bins
                       p(style="text-align: justify;","Set the filter parameters."),
                       numericInput(inputId='sliderrhapsodymtgene',
                                   label = 'Choose a MT Gene % < than ', value = 50, min = 5, max = 200),
                       
                       numericInput(inputId='sliderrhapsodyfeatures',
                                   label = 'Choose a number for minimum features > than', value = 200, min = 100, max = 5000),
                       hr(style="border-color: blue;"),
                       actionButton("submitrhapsody", strong("Submit Analysis")),
          ),
          mainPanel( width = 9,
                     tabsetPanel(
                       tabPanel(title = 'Preprocessing ',
                                plotlyOutput(outputId='rhapsodymtgene',height = "600px"),
                                h4(strong("Exporting the MT gene% plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_mtgene","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_mtgene","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_mtgene","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_mtgene','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodymtgenefilter',height = "600px"),
                                h4(strong("Exporting the MT gene% After Filter plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_mtgenefilter","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_mtgenefilter","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_mtgenefilter","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_mtgenefilter','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodyfeaturescatter',height = "600px"),
                                h4(strong("Exporting the Feature Scatter plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_featurescatter","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_featurescatter","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_featurescatter","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_featurescatter','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodyfeaturescatterfilter',height = "600px"),
                                h4(strong("Exporting the Feature Scatter plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_featurescatterfilter","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_featurescatterfilter","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_featurescatterfilter","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_featurescatterfilter','Download PNG'))
                                )
                       ),
                       tabPanel(title = 'Clustering Plots',
                                plotlyOutput(outputId='rhapsodyumap',height = "600px"),
                                h4(strong("Exporting the UMAP plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_umap","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_umap","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_umap","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_umap','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodytsne',height = "600px"),
                                h4(strong("Exporting the TSNE plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_tsne","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_tsne","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_tsne","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_tsne','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodypca',height = "600px"),
                                h4(strong("Exporting the PCA plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_ypca","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_ypca","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_ypca","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_ypca','Download PNG'))
                                )
                       ),
                       
                       tabPanel(title = 'Cell Type Annotation',
                                plotOutput(outputId='rhapsodyplotScoreHeatmap',height = "600px"),
                                h4(strong("Exporting the SingleR plotScoreHeatmap plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_plotScoreHeatmap","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_plotScoreHeatmap","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_plotScoreHeatmap","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_plotScoreHeatmap','Download PNG'))
                                ),
                                plotlyOutput(outputId='rhapsodyumapcelltype',height = "600px"),
                                h4(strong("Exporting the SingleR UMAP Cell type plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_umapcelltype","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_umapcelltype","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_umapcelltype","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_umapcelltype','Download PNG'))
                                )
                       ),
                       tabPanel(title = 'Merge and Remove batch effect',
                                plotlyOutput(outputId='rhapsodybatcheffect',height = "600px"),
                       ),
                       tabPanel(title = 'Finding doublets',
                                plotlyOutput(outputId='rhapsodydoublet',height = "600px"),
                                h4(strong("Exporting the Doublets plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_doublet","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_doublet","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_doublet","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_doublet','Download PNG'))
                                )
                       ),
                       tabPanel(title = 'Finding marker genes',
                                plotlyOutput(outputId='rhapsodymarkergenes',height = "1000px"),
                                h4(strong("Exporting the Markers genes plot")),
                                fluidRow(
                                  column(3,numericInput("width_png_markergenes","Width of PNG", value = 1600)),
                                  column(3,numericInput("height_png_markergenes","Height of PNG", value = 1200)),
                                  column(3,numericInput("resolution_PNG_markergenes","Resolution of PNG", value = 144)),
                                  column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_markergenes','Download PNG'))
                                )
                       ),
                       navbarMenu(title = 'Further Analysis',
                                  tabPanel(title = 'Pseudotime Analysis',
                                           plotOutput(outputId='rhapsodypseudotimelineage',height = "1000px"),
                                           h4(strong("Exporting the Pseudotime Analysis plot")),
                                           fluidRow(
                                             column(3,numericInput("width_png_pseudotimelineage","Width of PNG", value = 1600)),
                                             column(3,numericInput("height_png_pseudotimelineage","Height of PNG", value = 1200)),
                                             column(3,numericInput("resolution_PNG_pseudotimelineage","Resolution of PNG", value = 144)),
                                             column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_pseudotimelineage','Download PNG'))
                                           ),
                                           plotOutput(outputId='rhapsodypseudotimesampletag',height = "1000px"),
                                           h4(strong("Exporting the Pseudotime Analysis plot")),
                                           fluidRow(
                                             column(3,numericInput("width_png_pseudotimesampletag","Width of PNG", value = 1600)),
                                             column(3,numericInput("height_png_pseudotimesampletag","Height of PNG", value = 1200)),
                                             column(3,numericInput("resolution_PNG_pseudotimesampletag","Resolution of PNG", value = 144)),
                                             column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_pseudotimesampletag','Download PNG'))
                                           )
                                  ),
                                  tabPanel(title = 'Cell Communication',
                                           plotlyOutput(outputId='rhapsodycellcommunication',height = "600px"),
                                  )
                       ) # navbarMenu
                     ) #tabsetPanel 
          ) #mainPanel
        ) # sidebarLayout
      ) # fluidPage
    ) #tabItem
    # ================================================================================  
  ) #tabItems
) # dashboardBody


ui <- dashboardPage(
  header = dashHeader ,
  sidebar = dashsidebar,
  body = dashbody,
  title = 'Example dashboard',
  skin = 'blue'
)

# ======================================================================  Server =================================================================================================##

server <- shinyServer(function(input, output, session)
{
  vals = reactiveValues() # For the dowload
  ## ============================================= Graph panel results ====================================================== ##
  
  Datagraph <- reactive({switch(input$datasetgraph,"test-data" = test.data.graph(),"own" = own.data.graph())})
  
  test.data.graph <- reactive ({ 
    data(breast.TCGA) # from the mixomics package.
    mRna = data.frame(breast.TCGA$data.train$mrna)
    mRna$subtype = breast.TCGA$data.train$subtype
    mRna
  })
  
  own.data.graph <- reactive({
    if(is.null(input$filegraph)){
      return(NULL)
    }
    dataframe = readr::read_csv(input$filegraph$datapath)
    
  })
  
  observe({
    updateSelectInput(
      inputId = 'Vartoplot',
      session = session,
      label = 'Please choose a variable to plot',
      choices = names(Datagraph())
    )
  })
  
  observe({
    updateSelectInput(
      inputId = 'VarColor',
      session = session,
      label = 'Please choose a variable for color',
      choices = names(Datagraph()),
      selected = 'subtype'
    )
  })
  
  observe({
    updateSelectInput(
      inputId = 'Vartoplot1',
      session = session,
      label = 'Please choose a variable y',
      choices = names(Datagraph())
    )
  })
  
  observe({
    updateSelectInput(
      inputId = 'VarColor1',
      session = session,
      label = 'Please choose a variable x',
      choices = names(Datagraph()),
      selected = 'subtype'
    )
  })
  
  observe({
    updateSelectInput(
      inputId = 'Vartofill',
      session = session,
      label = 'Please choose a variable color',
      choices = c('NULL',names(Datagraph())),
      selected = 'subtype'
    )
  })
  #++++++++++++++++++++++++++++++++++++++++ Histogram 
  output$Histplot <- renderPlotly({
    Hist <- ggplot(Datagraph(), aes_string(x=input$Vartoplot, fill=input$VarColor)) +  geom_histogram(bins = input$histbins)
    Hist1 = Hist %>% ggplotly(tooltip = 'all')
    vals$Hist = Hist
  })
  # downloading PNG -----
  output$downloadPlotPNG_hist <- func_save_png(titlepng = "Histogram_", img = vals$Hist, width = input$width_png_hist, 
                                              height = input$height_png_hist, res = input$resolution_PNG_hist)
  
  #++++++++++++++++++++++++++++++++++++++++ density
  output$density <- renderPlotly({
    Dens <- ggplot(Datagraph(), aes_string(x=input$Vartoplot1, fill=input$VarColor1)) +  geom_density(fill='grey50')
    Dens1 = Dens %>% 
      ggplotly(tooltip = 'all') %>%
      layout(dragmode = "select")
    vals$densityplot = Dens
  })
  # downloading Density plot PNG -----
  output$downloadPlotPNG_densityplot <- func_save_png(titlepng = "Density_plot_", img = vals$densityplot, width = input$width_png_densityplot, 
                                                     height = input$height_png_densityplot, res = input$resolution_PNG_densityplot)
   
  #++++++++++++++++++++++++++++++++++++++++ Box plot
  output$boxplot <- renderPlotly({
    Boxp <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_boxplot() 
    Boxp1 = Boxp %>% 
      ggplotly(tooltip = 'all') 
    vals$Boxp = Boxp
  })
  # downloading Box plot PNG -----
  output$downloadPlotPNG_boxplot <- func_save_png(titlepng = "Box_plot_", img = vals$Boxp, width = input$width_png_boxplot, 
               height = input$height_png_boxplot, res = input$resolution_PNG_boxplot)
  
  #++++++++++++++++++++++++++++++++++++++++ Violin plot
  output$violinplot <- renderPlotly({
    Violin <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_violin() 
    Violin1 = Violin %>% 
      ggplotly(tooltip = 'all') 
    vals$Violin = Violin
  })
  # downloading Violin PNG -----
  output$downloadPlotPNG_violin <- func_save_png(titlepng = "Violin_plot_", img = vals$Violin, width = input$width_png_violin, 
                                                height = input$height_png_violin, res = input$resolution_PNG_violin)
   
  #++++++++++++++++++++++++++++++++++++++++ Linear plot
  output$linearplot <- renderPlotly({
    
    if(input$Vartofill == 'NULL'){
      Corrp <-  ggplot(Datagraph(), aes_string(input$VarColor1, input$Vartoplot1 )) + geom_point(position = "jitter") 
      Corrp1 = Corrp %>% 
        ggplotly(tooltip = 'all')
      vals$linearplot = Corrp
    }else{
      Corrp <-  ggplot(Datagraph(), aes_string(input$VarColor1, input$Vartoplot1, fill=input$Vartofill)) + geom_point(position = "jitter") #+ geom_smooth(method="lm", se = FALSE) 
      Corrp1 = Corrp %>% 
        ggplotly(tooltip = 'all')
      vals$linearplot = Corrp
    }
  })
  # downloading Linear plot PNG -----
  output$downloadPlotPNG_linearplot <- func_save_png(titlepng = "Linear_plot_", img = vals$linearplot, width = input$width_png_linearplot, 
                                                    height = input$height_png_linearplot, res = input$resolution_PNG_linearplot)
  
  #++++++++++++++++++++++++++++++++++++++++ Data table
  output$thetable <- DT::renderDataTable({
    DT::datatable(Datagraph(), filter = 'top', rownames = TRUE, options = list(scrollX = TRUE))
  },
  server = TRUE)
  
  output$summarythetable <- renderPrint({
    summary(Datagraph())
  })
  
  ## =========================================================================.  Differntial Panel results.  =============================================================================== #
  
  Datadiff <- reactive({switch(input$datasetdiff,"test-data" = test.data.diff(),"own" = own.data.diff())})
  
  test.data.diff <- reactive({
    Transcriptomics_data 
  })
  own.data.diff <- reactive({
    if(is.null(input$filediff)){
      return(NULL)
    }
    dataframe = readr::read_csv(input$filediff$datapath)
  })
  
  Diffdata <- reactive({
    Datadiff = data.frame(Datadiff())
    # add a column of NAs
    Datadiff$Direction <- "NO_Signif"
    # if log2Foldchange > 0.6 and pvalue < 0.05, set as "UP" 
    Datadiff$Direction[Datadiff$logFC > 0.6 & Datadiff$Pvalue < 0.05] <- "UP"
    # if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"
    Datadiff$Direction[Datadiff$logFC < -0.6 & Datadiff$Pvalue < 0.05] <- "DOWN"
    
    # Now write down the name of genes beside the points...
    # Create a new column "gene_name" to de, that will contain the name of genes differentially expressed (NA in case they are not)
    Datadiff$gene_name <- NA
    Datadiff$gene_name[Datadiff$Direction != "NO_Signif"] <- Datadiff$ID[Datadiff$Direction != "NO_Signif"]
    Diffdata <- Datadiff
  })
  
  output$number_of_points <- renderPrint({
    dat <- data.frame(Diffdata())
    dat <- dat[order(dat$Pvalue),]
    dat$logP <- -log10(dat$Pvalue)
    total <- as.numeric(dim(dat)[1])
    totalDown <- as.numeric(dim(dat[dat$Direction=='DOWN',])[1])
    totalNO <- as.numeric(dim(dat[dat$Direction=='NO_Signif',])[1])
    totalUP <- as.numeric(dim(dat[dat$Direction=='UP',])[1])
    
    cat('\n There are a total of ', total, ' where '  , totalDown, ' are dowregulated ', totalUP, ' are upregulated and ', totalNO, ' are none, ', 
        ' which represents ' ,round(totalNO/total*100,2),'% of the data',sep='')
  })
  
  output$Volcanoplot <- renderPlotly({
    Datadiff = data.frame(Diffdata())
    volcano_gplot <- ggplot(data=Datadiff, aes(x=logFC, y=-log10(Pvalue), col=Direction, label=gene_name)) + 
      geom_point() + 
      theme_minimal() +
      geom_text_repel() +
      scale_color_manual(values=c("blue", "black", "red")) +
      # Add vertical lines for log2FoldChange thresholds, and one horizontal line for the p-value threshold 
      geom_vline(xintercept=c(-0.6, 0.6), col="red") +  
      geom_hline(yintercept=-log10(0.05), col="red")
    volcano_gplotly <- volcano_gplot %>% 
      ggplotly(tooltip = 'all') 
    vals$volcano_gplot = volcano_gplot
  })
  # downloading Volcano plot PNG -----
  output$downloadPlotPNG_volcano <- func_save_png(titlepng = "Volcanoplot_", img = vals$volcano_gplot, width = input$width_png_volcano, 
                                                 height = input$height_png_volcano, res = input$resolution_PNG_volcano)
  # Filter data and do ID conversion
  type.of.data <- function () {
    
    dat <- Diffdata();
    
    
    dat <- dat[, c("ID","logFC","Pvalue", "Direction")]
    
    ID.conversion <- read.csv(system.file("extdata","uniprot.d.anno.210905.csv",package = "ggVolcanoR"))
    dat <- as.data.frame(dat)
    dat <- dat[order(dat$Pvalue),]
    rownames(dat) <- 1:dim(dat)[1]
    names(ID.conversion) <- c("Ensembl","Uniprot_human","UNIPROT","Chrom","Gene.Name","Biotype")
    
    dat.top <- merge(dat,ID.conversion,by.x="ID",by.y="Gene.Name", all.x=T)
    dat.top[is.na(dat.top)] <- "No_ID"
    names(dat.top) <- c("ID","logFC","Pvalue", "Direction","protein_atlas","UniProt_ID","UniProt_human","chrom","Biotype")
    type.of.data <- dat.top
    
  }
  
  #======
  output$summarytable <- DT::renderDataTable({
    top <- type.of.data()
    # dat <- data.frame(Diffdata())
    # SYMBOL_list <- as.data.frame(paste(dat$ID,"_",input$species,sep=""))
    # names(SYMBOL_list) <- "list"
    
    top$HGNC <- paste('<a href=https://www.genenames.org/tools/search/#!/?query=',top$ID,' target="_blank" class="btn btn-link"','>',top$ID,'</a>',sep="")
    top$GeneCards <- paste('<a href=https://www.genecards.org/cgi-bin/carddisp.pl?gene=',top$ID,' target="_blank" class="btn btn-link"','>',top$ID,'</a>',sep="")
    top$Protein_atlas <- paste('<a href=https://www.proteinatlas.org/',top$protein_atlas,' target="_blank" class="btn btn-link"','>',top$protein_atlas,'</a>',sep="")
   # top$Human_Uniprot <- paste('<a href=https://www.uniprot.org/uniprot/?query=',top$UniProt_human,' target="_blank" class="btn btn-link"','>',top$UniProt_human,"</a>", sep="")
    top$UniProt <- paste('<a href=https://www.uniprot.org/uniprot/?query=',top$UniProt_ID,' target="_blank" class="btn btn-link"','>',top$UniProt_ID,'</a>',sep="")
    top$Ensembl <- paste('<a href=https://www.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=',top$protein_atlas,' target="_blank" class="btn btn-link"','>',top$protein_atlas,'</a>',sep="")
    top <- top[,!names(top) %in% c("protein_atlas","UniProt_ID","UniProt_human","Gene.Name")]
    top <- top[order(top$Pvalue),]
    
    top <- DT::datatable(top, escape = FALSE, filter = 'top', options = list(scrollX = TRUE), rownames = FALSE) %>%
      formatStyle('logFC', 
                  backgroundColor = styleInterval(c(-0.6,0.6), c('#abd7eb', '#D2D2CF',"#ff6961")),
                  color = styleInterval(c(-0.6,0.6), c('#181A18', '#181A18', '#181A18')),
                  fontWeight = styleInterval(c(-0.6,0.6), c('bold', 'normal','bold'))) %>% 
      formatStyle('Pvalue',
                  backgroundColor = styleInterval(c(0.05), c('#181A18', '#D2D2CF')),
                  color = styleInterval(c(0.05), c('#d99058',  '#181A18')),
                  fontWeight = styleInterval(0.05, c('bold', 'normal'))
      )
  })
  
  output$summarytablecount <- DT::renderDataTable({
    dat <- data.frame(Diffdata())
    dat <- dat %>% dplyr::count(Direction)
    colnames(dat) <- c('Direction', 'Count')
    dat[nrow(dat) +1,] = c('Total', sum(dat$Count))
    
    DT::datatable(dat, options = list(scrollX = TRUE))
    
  })
  # Réagir lorsque l'utilisateur clique sur le bouton Soumettre
  observeEvent(input$diffsubmit, {
    dat <- data.frame(Diffdata())
    Geneup <- dat %>% 
      dplyr::arrange(desc(logFC)) %>%
      dplyr::slice(1:10)
    gene_list <- Geneup$gene_name
    # Exécuter l'analyse d'enrichissement de gènes avec enrichR
    enrichment_result <- enrichr(gene_list, dbs$libraryName)
    
    output$diffenrichplot <- renderPlotly({
      
      enrichment =  plotEnrich(enrichment_result[[input$diffenrich]], showTerms = 20, numChar = 50,
                               y = "Count", orderBy = "P.value", title = "Enrichment analysis by Enrichr",
                               xlab = "Enriched terms")
    })
    
  })
  ## =========================================================================.  Enrichment.  =============================================================================== #
  ##### Enrichment
  # Réagir lorsque l'utilisateur clique sur le bouton Soumettre
  observeEvent(input$submit, {
    # Extraire les gènes saisis par l'utilisateur et les séparer par des virgules
    user_genes <- unlist(strsplit(input$genes, ","))
  
    # Exécuter l'analyse d'enrichissement de gènes avec enrichR
    enrichment_result <- enrichr(user_genes, dbs$libraryName)
    
    # Afficher les résultats dans la sortie
    output$results <- renderPrint({
      head(enrichment_result[[input$databaseenrich]])
    })
    
    output$thetableenrich <- DT::renderDataTable({
      DT::datatable(enrichment_result[[input$databaseenrich]], filter = 'top', rownames = TRUE, options = list(scrollX = TRUE)) %>%
        formatStyle('P.value',
                    backgroundColor = styleInterval(c(0.05), c('#181A18', '#D2D2CF')),
                    color = styleInterval(c(0.05), c('#d99058',  '#181A18')),
                    fontWeight = styleInterval(0.05, c('bold', 'normal'))
        )
    },
    server = TRUE)
    
    output$gsea <- renderPlotly({
      
      enrichment =  plotEnrich(enrichment_result[[input$databaseenrich]], showTerms = 20, numChar = 50,
                               y = "Count", orderBy = "P.value", title = "Enrichment analysis by Enrichr",
                               xlab = "Enriched terms")
      vals$enrichment = enrichment
    })
    
    # downloading Enrichment plot PNG -----
    output$ downloadPlotPNG_enrichr <- func_save_png(titlepng = "Enrichment_Analysis_", img = vals$enrichment, width = input$width_png_enrichr, 
                                                    height = input$height_png_enrichr, res = input$resolution_PNG_enrichr)
  })
  
  output$enrichrdatabase <- DT::renderDataTable({
    DT::datatable(dbs, rownames = dbs$libraryName, options = list(scrollX = TRUE))
  },
  server = TRUE)
  ## =========================================================================.  Statistical Panel results.  =============================================================================== #
  
  # ===========================================================PCA
  Datastats <- reactive({switch(input$datasetstats,"test-data" = test.data.stats(),"own" = own.data.stats())})
  
  test.data.stats <- reactive ({ 
    mRna = data.frame(breast.TCGA$data.train$mrna)
  })
  
  own.data.stats <- reactive({
    if(is.null(input$filestats)){
      return(NULL)
    }
    dataframe = read.csv(input$filestats$datapath, row.names = 1)
    
  })
  #### def Metadata test
  subtype = data.frame(breast.TCGA$data.train$subtype)
  colnames(subtype) = 'subtype'
  rownames(subtype) = rownames(breast.TCGA$data.train$mrna)
  ####
  
  Metadastats <- reactive({switch(input$datasetstatsmetd,"test-metadata" = test.data.stats.metd(),"own-metadata" = own.data.stats.metd())})
  
  
  test.data.stats.metd <- reactive({
    subtype = data.frame(subtype)
  })
  
  own.data.stats.metd <- reactive({
    if(is.null(input$filestatsmetd)){
      return(NULL)
    }
    dataframe = read.csv(input$filestatsmetd$datapath, row.names = 1)
  })
  
  observe({
    updateSelectInput(
      inputId = 'Vartoplotstats',
      session = session,
      label = 'Please choose a metadata variable for annotations',
      choices = names(Metadastats())
    )
  })
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ PCA 
  pca <- reactive({
    Datastats <- as.matrix(Datastats())
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
  })
  
  # ======= PCA Ind  
  output$pcaind <- renderPlotly({
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pcafig = fviz_pca_ind(pca(), fill.ind = metadata[,1],  geom.ind = "point", 
                          pointshape=21,addEllipses = F,pointsize=4 )
    vals$pcafigind = pcafig
  })
  
  # downloading PCA Individuals PDF -----
  output$downloadPlotPDF_pcaind <- func_save_pdf(titlepdf = "PCA_Individuals_", img = vals$pcafigind, width = input$width_pcaind, 
                                       height = input$height_pcaind)
    
  # downloading PCA Individuals PNG -----
  output$downloadPlotPNG_pcaind <- func_save_png(titlepng = "PCA_Individuals_", img = vals$pcafigind, width = input$width_png_pcaind, 
                                         height = input$height_png_pcaind, res = input$resolution_PNG_pcaind)
  
  # ======= PCA biplot
  output$pcabiplot <- renderPlotly({
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
  
    pcabiplot = fviz_pca_biplot(pca(), fill.ind = metadata[,1], geom.ind = "point", 
                                pointshape=10,addEllipses = T,pointsize=2)
    vals$pcabiplot = pcabiplot
  })
  
  # downloading PCA Biplot PNG -----
  output$downloadPlotPNG_biplot <- func_save_png(titlepng = "PCA_Biplot_", img = vals$pcabiplot, width = input$width_png_biplot, 
                                                 height = input$height_png_biplot, res = input$resolution_PNG_biplot)
  
  # ======= PCA component
  output$pcacomp <- renderPlotly({
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
  
    pcacomp = fviz_eig(pca())
    vals$pcacomp = pcacomp
  })
  
  # downloading PCA Component PNG -----
  output$downloadPlotPNG_scree <- func_save_png(titlepng = "PCA_Component_", img = vals$pcacomp, width = input$width_png_scree, 
                                                height = input$height_png_scree, res = input$resolution_PNG_scree)
   
  # ======= PCA variable importance
  output$variableimp <- renderPlotly({
    pca = pca()
    
    loadings <- data.frame(pca$var$coord[1:10,1:5])
    loadings$Symbol <- row.names(loadings)
    loadings <- gather(loadings, 'Component', 'Weight',
                       -Symbol)
    colors <- list()
    for (i in 1:nrow(loadings)){
      colors= ifelse(loadings$Weight > 0,'Positive','Negative')
    }
    pcavar <- ggplot(loadings, aes(x=Symbol, y=Weight)) +
      geom_bar(stat='identity', aes(fill = colors)) +
      facet_grid(Component ~
                   ., scales='free_y') +
      labs(title = "Gene component association",
           x = "Gene symbol",
           y = "Gene weight for each component",
           colour = "Up/Down") +
      theme(axis.text.x = element_text(angle = 90))
    theme(plot.title = element_text(hjust = 0.5))
    plot <- pcavar %>% 
      ggplotly(tooltip = 'all') 
    vals$pcavar = pcavar
  })
  
  # downloading PCA Gene Importances PNG -----
  output$downloadPlotPNG_genepca <- func_save_png(titlepng = "PCA_Importance_Genes_", img = vals$pcavar, width = input$width_png_genepca, 
                                                  height = input$height_png_genepca, res = input$resolution_PNG_genepca)
  
  # Stats Data Table 
  output$thetablestats <- DT::renderDataTable({
    DT::datatable(Datastats(), filter = 'top', options = list(scrollX = TRUE))
  },
  server = TRUE)
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Clustring
  
  Datastatsclust <- reactive({switch(input$datasetstatsclust,"test-data" = test.data.statsclust(),"own" = own.data.statsclust())})
  
  test.data.statsclust <- reactive ({ 
    mRna = data.frame(breast.TCGA$data.train$mrna)
  })
  
  own.data.statsclust <- reactive({
    if(is.null(input$filestatsclust)){
      return(NULL)
    }
    dataframe = read.csv(input$filestatsclust$datapath, row.names = 1)
    
  })
  #### def Metadata test
  subtype = data.frame(breast.TCGA$data.train$subtype)
  colnames(subtype) = 'subtype'
  rownames(subtype) = rownames(breast.TCGA$data.train$mrna)
  ####
  
  Metadastatsclust <- reactive({switch(input$datasetstatsmetdclust,"test-metadata" = test.data.stats.metdclust(),"own-metadata" = own.data.stats.metdclust())})
  
  
  test.data.stats.metdclust <- reactive({
    subtype = data.frame(subtype)
  })
  
  own.data.stats.metdclust <- reactive({
    if(is.null(input$filestatsmetdclust)){
      return(NULL)
    }
    dataframe = read.csv(input$filestatsmetdclust$datapath, row.names = 1)
  })
  
  observe({
    updateSelectInput(
      inputId = 'Vartoplotstatsclust',
      session = session,
      label = 'Please choose a metadata variable for annotations',
      choices = names(Metadastatsclust())
    )
  })
  
  ##========== Heatmap
  
  output$heatmap <- renderPlot({
    Datastats <- as.matrix(Datastatsclust())
    heatmapplot = heatmap.2(Datastats, col = greenred(256), scale="column", margins=c(5,10), density="density", xlab = "Gene Names", 
                            ylab= "Samples ID", main = " ",breaks=seq(-1.5,1.5,length.out=257))
    
    vals$heatmapplot = heatmapplot
  })
  # downloading Heatmap PNG -----
  output$downloadPlotPNG_heatmap <- func_save_png(titlepng = "Heatmap_", img = vals$heatmapplot, width = input$width_png_heatmap, 
                                                  height = input$height_png_heatmap, res = input$resolution_PNG_heatmap)
    
  output$missmap <- renderPlot({
    missmap(Datastatsclust())
  })
  #++++++++++++++++++++++++++++++++++++++++++++++++++ Kmeans
  
  kmeanss <- reactive({
    kmeanss <- kmeans(scale(Datastatsclust()), input$kmeansbins, nstart = 25)
  })
  ##========== Kmeans
  output$kmeanscluster <- renderPlotly({
    # Clustering K-means montrant le groupe de chaque individu
    kmeanscluster = fviz_cluster(kmeanss(), data = Datastatsclust(),
                                 geom = "point",
                                 ellipse.type = "convex", 
                                 ggtheme = theme_bw()
    )
    vals$kmeanscluster = kmeanscluster
  })
  # downloading Kmeans clusters PNG -----
  output$downloadPlotPNG_kmeanscluster <- func_save_png(titlepng = "Kmeans_clusters_", img = vals$kmeanscluster, width = input$width_png_kmeanscluster, 
                                                        height = input$height_png_kmeanscluster, res = input$resolution_PNG_kmeanscluster)
  ##========== Kmeana annotate
  output$kmeansclusterannot <- renderPlotly({
    # Réduction de dimension en utilisant l'ACP
    res.pca <- prcomp(Datastatsclust(),  scale = TRUE)
    # Coordonnées des individus
    ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
    # Ajouter les clusters obtenus à l'aide de l'algorithme k-means
    ind.coord$cluster <- factor(kmeanss()$cluster)
    # Ajouter les groupes d'espèces issues du jeu de données initial
    metadata = Metadastatsclust() %>% dplyr::select(input$Vartoplotstatsclust)
    ind.coord$metadata <- metadata[,1]
    
    # Pourcentage de la variance expliquée par les dimensions
    eigenvalue <- round(get_eigenvalue(res.pca), 1)
    variance.percent <- eigenvalue$variance.percent
    
    #Ajoutez le centroid des clusters en utilisant la fonction R stat_mean() [ggpubr]
    kmeansclusterannot = ggscatter(
      ind.coord, x = "Dim.1", y = "Dim.2", 
      color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "convex",
      shape = "metadata", size = 1.5,  legend = "right", ggtheme = theme_bw(),
      xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
      ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
    ) +
      stat_mean(aes(color = cluster), size = 4)
    vals$kmeansclusterannot = kmeansclusterannot
  })
  
  # downloading Kmeans clusters annotation PNG -----
  output$downloadPlotPNG_kmeansclusterannot <- func_save_png(titlepng = "Kmeans_clustrs_annotated_", img = vals$kmeansclusterannot, width = input$width_png_kmeansclusterannot, 
                                                             height = input$height_png_kmeansclusterannot, res = input$resolution_PNG_kmeansclusterannot)
  
  output$kmeanselbow <- renderPlotly({
    kmeanselbow = fviz_nbclust(Datastatsclust(), kmeans, method = "wss") +
      geom_vline(xintercept = input$kmeansbins, linetype = 2)+
      labs(subtitle = "Elbow method") 
    vals$kmeanselbow = kmeanselbow
  })
  
  # downloading Kmeans Elbow plot PNG -----
  output$downloadPlotPNG_kmeanselbow <- func_save_png(titlepng = "Kmeans_Elbow_method_", img = vals$kmeanselbow, width = input$width_png_kmeanselbow, 
                                                      height = input$height_png_kmeanselbow, res = input$resolution_PNG_kmeanselbow)
   
  #++++++++++++
  output$thetablestatsclust <- DT::renderDataTable({
    DT::datatable(Datastatsclust(), filter = 'top', options = list(scrollX = TRUE))
  },
  server = TRUE)
  ## =======================================================================================. IGV =========================================================================================================#
  observeEvent(input$genomeChooser, ignoreInit=FALSE, {
    newGenome <- input$genomeChooser
    genomeSpec <- parseAndValidateGenomeSpec(genomeName=newGenome,  initialLocus="all")
    output$igvShiny <- renderIgvShiny(
      igvShiny(genomeSpec)
    )
  })
  ## =======================================================================================. Rhapsody =========================================================================================================#
  Datastatrhapsody <- reactive({switch(input$datasetrhapsody,"test-data" = test.data.rhapsody(),"own" = own.data.rhapsody())})
  
  test.data.rhapsody <- reactive ({ 
    readRDS("Protocol-rerun_Seurat.rds")
  })
  
  own.data.rhapsody <- reactive({
    if(is.null(input$filerhapsody)){
      return(NULL)
    }
    dataframe = readRDS(input$filerhapsody$datapath)
    
  })
  # Réagir lorsque l'utilisateur clique sur le bouton Soumettre analyse
  observeEvent(input$submitrhapsody, {
    
  demo_seurat <- reactive({
    demo_seurat <- func_get_AbSeq(demo_seurat = Datastatrhapsody())
    demo_seurat$smk <- demo_seurat$Sample_Name
    demo_seurat <- func_quick_process(demo_seurat)
  })
  
  # filter out cells with MT genes percentage > 50 (%) and 
  # cells with low nFeature_RNA
  subset_demo_seurat <- reactive({
    subset_demo_seurat <- subset(demo_seurat(), 
                                 subset = percent.mt < input$sliderrhapsodymtgene & 
                                   nFeature_RNA > input$sliderrhapsodyfeatures, 
                                 invert = F)
    subset_demo_seurat <- func_quick_process(subset_demo_seurat)
    subset_demo_seurat <- func_get_annotation(subset_demo_seurat)
  })
  # =============================. QC plots – check mitochondrial gene percentages
  output$rhapsodymtgene <- renderPlotly({
    
    p <- Seurat::VlnPlot(demo_seurat(), 
                         features = "percent.mt", 
                         group.by = "seurat_clusters") + 
      Seurat::NoLegend() + 
      ggtitle("MT Gene %")
    p
    vals$mtgene <- p
  })
  output$rhapsodymtgenefilter <- renderPlotly({
    
    p2 <- Seurat::VlnPlot(subset_demo_seurat(), 
                          features = "percent.mt", 
                          group.by = "seurat_clusters") + 
      Seurat::NoLegend() + 
      ggtitle("MT Gene % After Filter")
    p2
    
    vals$mtgenefilter <- p2
  })
  # downloading QC plots – check mitochondrial gene percentages PNG -----
  # downloading QC plots – check mitochondrial gene percentages PNG -----
  output$downloadPlotPNG_mtgene <- func_save_png(titlepng = "MT_Gene_%_", img = vals$mtgene, width = input$width_png_mtgene, 
                                                  height = input$height_png_mtgene, res = input$resolution_PNG_mtgene)
  # downloading QC plots – check mitochondrial gene percentages after filtering PNG -----
  output$downloadPlotPNG_mtgenefilter <- func_save_png(titlepng = "MT_Gene_%_After_Filter_", img = vals$mtgenefilter, width = input$width_png_mtgenefilter, 
                                                 height = input$height_png_mtgenefilter, res = input$resolution_PNG_mtgenefilter)
  
  # =================================. Feature Scatter 
  output$rhapsodyfeaturescatter <- renderPlotly({
    p3 <- Seurat::FeatureScatter(demo_seurat(), 
                                 feature1 = "nCount_RNA", 
                                 feature2 = "nFeature_RNA", 
                                 group.by = "seurat_clusters") + 
      scale_x_log10() +
      scale_y_log10() +
      ggtitle("Feature Scatter plot")
    
    p3
    vals$featurescatter <- p3
  })
  
  output$rhapsodyfeaturescatterfilter <- renderPlotly({
    p4 <- Seurat::FeatureScatter(subset_demo_seurat(), 
                                 feature1 = "nCount_RNA", 
                                 feature2 = "nFeature_RNA", 
                                 group.by = "seurat_clusters") + 
      scale_x_log10() +
      scale_y_log10() +
      ggtitle("Feature Scatter plot After Filter")
    
    p4
    vals$featurescatterfilter <- p4
  })
  # downloading Feature Scatter plot PNG -----
  output$downloadPlotPNG_featurescatter <- func_save_png(titlepng = "Feature_Scatter_plot_", img = vals$featurescatter, width = input$width_png_featurescatter, 
                                                 height = input$height_png_featurescatter, res = input$resolution_PNG_featurescatter)
  # downloading Feature Scatter after filtering plot PNG -----
  output$downloadPlotPNG_featurescatterfilter <- func_save_png(titlepng = "Feature_Scatter_plot_", img = vals$featurescatterfilter, width = input$width_png_featurescatterfilter, 
                                                         height = input$height_png_featurescatterfilter, res = input$resolution_PNG_featurescatterfilter)
  
  # ===================================. clustering plots
  output$rhapsodyumap <- renderPlotly({
    p5 <- Seurat::DimPlot(subset_demo_seurat(), 
                          reduction = "umap", 
                          group.by = "seurat_clusters") + 
      ggtitle("UMAP Plot")
    vals$umap <- p5
  })
  # downloading Rhapsody Umap plot PNG -----
  output$downloadPlotPNG_umap <- func_save_png(titlepng = "Umap_plot_", img = vals$umap, width = input$width_png_umap, 
                                               height = input$height_png_umap, res = input$resolution_PNG_umap)
  output$rhapsodytsne <- renderPlotly({
    p6 <- Seurat::DimPlot(subset_demo_seurat(), 
                          reduction = "tsne", 
                          group.by = "seurat_clusters") + 
      ggtitle("TSNE Plot")
    vals$tsne <- p6
  })
  # downloading Rhapsody TSNE plot PNG -----
  output$downloadPlotPNG_tsne <- func_save_png(titlepng = "Tsne_plot_", img = vals$tsne, width = input$width_png_tsne, 
                                               height = input$height_png_tsne, res = input$resolution_PNG_tsne)
  
  output$rhapsodypca <- renderPlotly({
    p7 <- Seurat::DimPlot(subset_demo_seurat(), 
                          reduction = "pca", 
                          group.by = "seurat_clusters") + 
      ggtitle("PCA Plot")
    vals$rhapsodypca <- p7
  })
  # downloading Rhapsody PCA plot PNG -----
  output$downloadPlotPNG_ypca <- func_save_png(titlepng = "PCA_plot_", img = vals$rhapsodypca, width = input$width_png_ypca, 
                                               height = input$height_png_ypca, res = input$resolution_PNG_ypca)
  
  # ============================================================================. SingleR plots
  output$rhapsodyplotScoreHeatmap <- renderPlot({
    subset_demo_seurat <- subset_demo_seurat()
    p_cell_1 <- plotScoreHeatmap(subset_demo_seurat@misc$SingleR_results,
                                 show_colnames = F)
    p_cell_1
    vals$plotScoreHeatmap <- p_cell_1
  })
  # downloading SingleR plotScoreHeatmap plot PNG -----
  output$downloadPlotPNG_plotScoreHeatmap <- func_save_png(titlepng = "SingleR_plotScoreHeatmap_", img = vals$plotScoreHeatmap, width = input$width_png_plotScoreHeatmap, 
                                               height = input$height_png_plotScoreHeatmap, res = input$resolution_PNG_plotScoreHeatmap)
  
  output$rhapsodyumapcelltype <- renderPlotly({
    # Display cells in UMAP plot
    p_cell_2 <- Seurat::DimPlot(subset_demo_seurat(),
                                group.by = "cell_type") +
      ggtitle("SINGLER UMAP CELL TYPE")
    vals$umapcelltype <- p_cell_2
  })
  # downloading SingleR UMAP Cell type plot PNG -----
  output$downloadPlotPNG_umapcelltype <- func_save_png(titlepng = "SingleR_UMAP_cell_type_", img = vals$umapcelltype, width = input$width_png_umapcelltype, 
                                                           height = input$height_png_umapcelltype, res = input$resolution_PNG_umapcelltype)
  
  # ============================================================================. Find doublets
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
  # Run find doublets func
  subset_demo_seurat1 <- reactive({
    subset_demo_seurat1 <- func_get_doublets(subset_demo_seurat(),
                                            pc = 1:15)
  })
  output$rhapsodydoublet <- renderPlotly({
    # Visualize the result
    p_cell_3 <- DimPlot(subset_demo_seurat1(), 
                        group.by = "doublet_check") + 
      ggtitle("Doublet Check Plot")
    vals$doublet <- p_cell_3
  })
  # downloading doublet rates plot PNG -----
  output$downloadPlotPNG_doublet <- func_save_png(titlepng = "Doublet_Check_plot_", img = vals$doublet, width = input$width_png_doublet, 
                                                       height = input$height_png_doublet, res = input$resolution_PNG_doublet)
  
  # ============================================================================. Finding marker genes
  output$rhapsodymarkergenes <- renderPlotly({
    # use function to get marker genes
    subset_demo1_DGEs <- func_get_marker_genes(subset_demo_seurat1(),
                                               p_adj_cutoff = 0.05,
                                               log2FC_cutoff = 1,
                                               view_top_X_genes = 5) 
    
    # visualise top genes on dotplot
    p_cell_4 <- DotPlot(subset_demo_seurat1(), 
                         features = subset_demo1_DGEs$top_cell_gene, 
                         group.by = "cell_type") + 
      coord_flip() +
      RotatedAxis() +
      ggtitle("Marker Genes plot")
    vals$markergenes <- p_cell_4
  })
  # downloading Marker Genes plot PNG -----
  output$downloadPlotPNG_markergenes <- func_save_png(titlepng = "Marker_Genes_plot_", img = vals$markergenes, width = input$width_png_markergenes, 
                                                  height = input$height_png_markergenes, res = input$resolution_PNG_markergenes)
  
  # ============================================================================. Pseudotime Analysis
  subset_demo_slingshot_1 <- reactive({
    # use function to get results
    subset_demo_slingshot_1 <- func_slingshot(subset_demo_seurat1())
  })
  # ++++++++++++++++++++++ Pseudotime
  output$rhapsodypseudotimelineage <- renderPlot({
   
    subset_demo_seurat_1 <- subset_demo_seurat1()
    
    pt_lineages <- slingshot::slingPseudotime(subset_demo_slingshot_1())
    
    # add Slingshot results to the input Seurat object
    lineages <- sapply(slingLineages(colData(subset_demo_slingshot_1())$slingshot), 
                       paste, 
                       collapse = " -> ")
    
    subset_demo_seurat_1@meta.data[lineages] <- pt_lineages
    # visualization
    
    # display every lineage pseudotime
    name_lineage <- colnames(subset_demo_seurat_1@meta.data)[grepl("->",
                                                                   colnames(subset_demo_seurat_1@meta.data))]
    
    p_ss_1 <- list()
    
    Idents(subset_demo_seurat_1) <- "smk"
    
    for (i in name_lineage) {
      
      p_ss_1[[i]] <- Seurat::FeaturePlot(subset(subset_demo_seurat_1, 
                                                idents = c("SampleTag01_hs", 
                                                           "SampleTag02_hs")),  
                                         features = i, split.by = "smk") & 
        theme(legend.position="top") &
        scale_color_viridis_c() 
    }
    
    vals$pseudotimelineage <- wrap_plots(p_ss_1, 
                                         ncol = 1)
    
    wrap_plots(p_ss_1, 
               ncol = 1)
  })
  # downloading Pseudotime lineage plot PNG -----
  output$downloadPlotPNG_pseudotimelineage <- func_save_png(titlepng = "Pseudotime_lineage_plot_", img = vals$pseudotimelineage, width = input$width_png_pseudotimelineage, 
                                                            height = input$height_png_pseudotimelineage, res = input$resolution_PNG_pseudotimelineage)
  
  output$rhapsodypseudotimesampletag <- renderPlot({
    subset_demo_seurat_1 <- subset_demo_seurat1()
    
    pt_lineages <- slingshot::slingPseudotime(subset_demo_slingshot_1())
    
    # add Slingshot results to the input Seurat object
    lineages <- sapply(slingLineages(colData(subset_demo_slingshot_1())$slingshot), 
                       paste, 
                       collapse = " -> ")
    
    subset_demo_seurat_1@meta.data[lineages] <- pt_lineages
    
    # display every lineage pseudotime
    name_lineage <- colnames(subset_demo_seurat_1@meta.data)[grepl("->",
                                                                   colnames(subset_demo_seurat_1@meta.data))]
    
    p_ss_celltype <- list()
    
    Idents(subset_demo_seurat_1) <- "smk"
    
    for (i in name_lineage) {
      
      p_ss_celltype[[i]] <- ggplot(subset(subset_demo_seurat_1, 
                                          idents = c("Multiplet", 
                                                     "Undetermined"), 
                                          invert = T)@meta.data, 
                                   aes(x = .data[[i]], 
                                       y = cell_type, 
                                       colour = cell_type)) +
        geom_point() +
        geom_jitter(width = 0.1, 
                    height = 0.2) +
        theme_gray() +
        theme(legend.position = "none") +
        facet_grid(. ~ smk )
    }
    
    vals$pseudotimesampletag <- wrap_plots(p_ss_celltype, 
                                           ncol = 2)
    
    wrap_plots(p_ss_celltype, 
               ncol = 2)
  })
  # downloading Pseudotime Sample tag plot PNG -----
  output$downloadPlotPNG_pseudotimesampletag <- func_save_png(titlepng = "Pseudotime_lineage_plot_", img = vals$pseudotimesampletag, width = input$width_png_pseudotimesampletag, 
                                                              height = input$height_png_pseudotimesampletag, res = input$resolution_PNG_pseudotimesampletag)
  
  }) # submitrhapsody 
  ## =======================================================================================. End Server =========================================================================================================#
  # This are for the server close
})

# =======================================================================================. App. ============================================================================================================#

shinyApp(ui = ui, server = server)

# =======================================================================================. End. ============================================================================================================#