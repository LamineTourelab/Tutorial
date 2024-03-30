library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)
library(ggrepel)
library(plotly)
library(gridExtra)
library(tidyr)
library(tidyverse)
library(dplyr)
library(factoextra) # for pca visualization 
library(ggpubr)
library(NbClust) # Kmeans best cluster number
library(FactoMineR)  # PCA
library(enrichR) #Enrichment analysis
library(gplots) # Use heatmap.2
library(corrplot)
library(mixOmics) # for the breast cancer dataset
library(Amelia) # for missing values visualization
library(igvShiny)
library(GenomicAlignments)
library(rtracklayer)
library(shinymanager)


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
                                     class = "dropdown"))
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
      icon = icon('dna', style = "color:#E87722"))
    
    
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
                                     plotlyOutput(outputId='pca',height = "600px"),
                                     h4(strong("Exporting the Individuals PCA plot")),
                                     fluidRow(
                                       
                                       column(3,numericInput("width", "Width of PDF", value=10)),
                                       column(3,numericInput("height", "Height of PDF", value=8)),
                                       column(3),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlot','Download PDF'))
                                     ),
                                     
                                     fluidRow(
                                       column(3,numericInput("width_png","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG","Resolution of PNG", value = 144)),
                                       column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG','Download PNG'))
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
                                       column(3,numericInput("width_png","Width of PNG", value = 1600)),
                                       column(3,numericInput("height_png","Height of PNG", value = 1200)),
                                       column(3,numericInput("resolution_PNG","Resolution of PNG", value = 144)),
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
            fluidRow(
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
                           # Placeholder for input selection
                           fluidRow(
                             column(6, selectInput(inputId='Vartoplotdiff',label = 'Waiting for data plot',choices = NULL )),
                             column(6, checkboxGroupInput(inputId='Vardatabasediff',label = 'Choose database',choices = NULL ))
                           )
              ),
              tabBox( width = 10,
                      tabPanel(title='Volcano plot',
                               textOutput("number_of_points"),
                               plotlyOutput(outputId = 'Volcanoplot',height = "600px"),
                               h4(strong("Exporting the Vocano plot")),
                               fluidRow(
                                 column(3,numericInput("width_png_volcano","Width of PNG", value = 1600)),
                                 column(3,numericInput("height_png_volcano","Height of PNG", value = 1200)),
                                 column(3,numericInput("resolution_PNG_volcano","Resolution of PNG", value = 144)),
                                 column(3,style = "margin-top: 25px;",downloadButton('downloadPlotPNG_volcano','Download PNG'))
                               )
                      ),
                      tabPanel(title='Summary Table',
                               #Placeholder for plot
                               h2("Data table:"),
                               DT::dataTableOutput(outputId = 'summarytable'),
                               h2("The summary count table:"),
                               DT::dataTableOutput(outputId = 'summarytablecount')
                      ),
                      tabPanel(title='Gene Set Enrichment',
                               
                      )
              )
            )
            
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
      
    )#tabItem
    # ==================================
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
  #++++++++++++++++++++ Histogram 
  output$Histplot <- renderPlotly({
    Hist <- ggplot(Datagraph(), aes_string(x=input$Vartoplot, fill=input$VarColor)) +  geom_histogram(bins = input$histbins)
    Hist1 = Hist %>% ggplotly(tooltip = 'all')
    vals$Hist = Hist
  })
  # downloading PNG -----
  output$downloadPlotPNG_hist <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Histogram_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_hist, height = input$height_png_hist, res = input$resolution_PNG_hist)
      grid.arrange(vals$Hist)
      dev.off()}
  )
  #++++++++++++++++++++ density
  output$density <- renderPlotly({
    Dens <- ggplot(Datagraph(), aes_string(x=input$Vartoplot1, fill=input$VarColor1)) +  geom_density(fill='grey50')
    Dens1 = Dens %>% 
      ggplotly(tooltip = 'all') %>%
      layout(dragmode = "select")
    vals$densityplot = Dens
  })
  # downloading PNG -----
  output$downloadPlotPNG_densityplot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Density_plot_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_densityplot, height = input$height_png_densityplot, res = input$resolution_PNG_densityplot)
      grid.arrange(vals$densityplot)
      dev.off()}
  )
  #++++++++++++++++++++ Box plot
  output$boxplot <- renderPlotly({
    Boxp <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_boxplot() 
    Boxp1 = Boxp %>% 
      ggplotly(tooltip = 'all') 
    vals$Boxp = Boxp
  })
  # downloading PNG -----
  output$downloadPlotPNG_boxplot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Box_plot_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_boxplot, height = input$height_png_boxplot, res = input$resolution_PNG_boxplot)
      grid.arrange(vals$Boxp)
      dev.off()}
  )
  #++++++++++++++++++++ Violin plot
  output$violinplot <- renderPlotly({
    Violin <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_violin() 
    Violin1 = Violin %>% 
      ggplotly(tooltip = 'all') 
    vals$Violin = Violin
  })
  # downloading PNG -----
  output$downloadPlotPNG_violin <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Violin_plot_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_violin, height = input$height_png_violin, res = input$resolution_PNG_violin)
      grid.arrange(vals$Violin)
      dev.off()}
  )
  #++++++++++++++++++++ Linear plot
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
  # downloading PNG -----
  output$downloadPlotPNG_linearplot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Linear_plot_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_linearplot, height = input$height_png_linearplot, res = input$resolution_PNG_linearplot)
      grid.arrange(vals$linearplot)
      dev.off()}
  )
  #++++++++++++++++++++ Data table
  output$thetable <- DT::renderDataTable({
    DT::datatable(Datagraph(), rownames = TRUE, options = list(scrollX = TRUE))
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
  
  # 
  # observe({
  #   updateSelectInput(
  #     inputId = 'Vartoplotdiff',
  #     session = session,
  #     label = 'Please choose a variable',
  #     choices = dbs$libraryName
  #   )
  # })
  
  output$number_of_points <- renderPrint({
    dat <- as.data.frame(Datadiff())
    dat$Direction <- "NO"
    dat$Direction[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$Direction[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$Direction != "NO"] <- dat$ID[dat$Direction != "NO"]
    dat <- dat[order(dat$Pvalue),]
    dat$logP <- -log10(dat$Pvalue)
    total <- as.numeric(dim(dat)[1])
    totalDown <- as.numeric(dim(dat[dat$Direction=='DOWN',])[1])
    totalNO <- as.numeric(dim(dat[dat$Direction=='NO',])[1])
    totalUP <- as.numeric(dim(dat[dat$Direction=='UP',])[1])
    
    cat('\n There are a total of ', total, ' where '  , totalDown, ' are dowregulated ', totalUP, ' are upregulated and ', totalNO, ' are none, ', 
        ' which represents ' ,round(totalNO/total*500,2),'% of the data',sep='')
  })
  
  output$Volcanoplot <- renderPlotly({
    
    Datadiff = data.frame(Datadiff())
    # add a column of NAs
    Datadiff$Direction <- "NO"
    # if log2Foldchange > 0.6 and pvalue < 0.05, set as "UP" 
    Datadiff$Direction[Datadiff$logFC > 0.6 & Datadiff$Pvalue < 0.05] <- "UP"
    # if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"
    Datadiff$Direction[Datadiff$logFC < -0.6 & Datadiff$Pvalue < 0.05] <- "DOWN"
    
    # Now write down the name of genes beside the points...
    # Create a new column "gene_name" to de, that will contain the name of genes differentially expressed (NA in case they are not)
    Datadiff$gene_name <- NA
    Datadiff$gene_name[Datadiff$Direction != "NO"] <- Datadiff$ID[Datadiff$Direction != "NO"]
    
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
  # downloading PNG -----
  output$downloadPlotPNG_volcano <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Volcanoplot_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_volcano, height = input$height_png_volcano, res = input$resolution_PNG_volcano)
      grid.arrange(vals$volcano_gplot)
      dev.off()}
  )
  #======
  output$summarytable <- DT::renderDataTable({
    dat <- data.frame(Datadiff())
    dat$Direction <- "NO"
    dat$Direction[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$Direction[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$Direction != "NO"] <- dat$ID[dat$Direction != "NO"]
    
    dat <- DT::datatable(dat, options = list(scrollX = TRUE))
    dat
  })
  
  output$summarytablecount <- DT::renderDataTable({
    dat <- data.frame(Datadiff())
    dat$Direction <- "NO"
    dat$Direction[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$Direction[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$Direction != "NO"] <- dat$ID[dat$Direction != "NO"]
    dat <- dat %>% dplyr::count(Direction)
    DT::datatable(dat, options = list(scrollX = TRUE))
  })
  
  ##### Enrichment
  # Réagir lorsque l'utilisateur clique sur le bouton Soumettre
  observeEvent(input$submit, {
    # Extraire les gènes saisis par l'utilisateur et les séparer par des virgules
    user_genes <- unlist(strsplit(input$genes, ","))
    
    # dbs <- c("GO_Molecular_Function_2015", "GO_Cellular_Component_2015", "GO_Biological_Process_2015",
    #          "Reactome_2015", "Reactome_2016", "OMIM_Disease", "MSigDB_Oncogenic_Signatures", "KEGG_2015",
    #          "KEGG_2016", "GO_Biological_Process_2018", "Human_Phenotype_Ontology", "Cancer_Cell_Line_Encyclopedia",
    #          "RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO")
    
    # Exécuter l'analyse d'enrichissement de gènes avec enrichR
    enrichment_result <- enrichr(user_genes, dbs$libraryName)
    
    # Afficher les résultats dans la sortie
    output$results <- renderPrint({
      head(enrichment_result[[input$databaseenrich]])
    })
    
    output$enrichrdatabase <- DT::renderDataTable({
      DT::datatable(dbs, rownames = dbs$libraryName, options = list(scrollX = TRUE))
    },
    server = TRUE)
    
    output$thetableenrich <- DT::renderDataTable({
      DT::datatable(enrichment_result[[input$databaseenrich]], rownames = TRUE, options = list(scrollX = TRUE))
    },
    server = TRUE)
    
    output$gsea <- renderPlotly({
      
      enrichment =  plotEnrich(enrichment_result[[input$databaseenrich]], showTerms = 20, numChar = 50,
                               y = "Count", orderBy = "P.value", title = "Enrichment analysis by Enrichr",
                               xlab = "Enriched terms")
      vals$enrichment = enrichment
    })
    
    # downloading PNG -----
    output$ downloadPlotPNG_enrichr <- downloadHandler(
      filename = function() {
        x <- gsub(":", ".", Sys.Date())
        paste("Enrichment_Analysis_",input$title, gsub("/", "-", x), ".png", sep = "")
      },
      content = function(file) {
        
        png(file, width = input$width_png_enrichr, height = input$height_png_enrichr, res = input$resolution_PNG_enrichr)
        grid.arrange(vals$enrichment)
        dev.off()}
    )
  })
  
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
  
  # ======= PCA Ind  
  output$pca <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    pcafig = fviz_pca_ind(pca, fill.ind = metadata[,1],  geom.ind = "point", 
                          pointshape=21,addEllipses = F,pointsize=4 )
    vals$pcafig = pcafig
  })
  
  # downloading PDF -----
  output$downloadPlot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("PCA_Individuals_",input$title, gsub("/", "-", x), ".pdf", sep = "")
    },
    content = function(file) {
      pdf(file, width=input$width,height=input$height, onefile = FALSE) # open the pdf device
      grid.arrange(vals$pcafig)
      dev.off()},
    
    contentType = "application/pdf"
    
  )
  # downloading PNG -----
  output$downloadPlotPNG <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("PCA_Individuals_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png, height = input$height_png, res = input$resolution_PNG)
      grid.arrange(vals$pcafig)
      dev.off()}
  )
  # ======= PCA biplot
  output$pcabiplot <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    pcabiplot = fviz_pca_biplot(pca, fill.ind = metadata[,1], geom.ind = "point", 
                                pointshape=10,addEllipses = T,pointsize=2)
    vals$pcabiplot = pcabiplot
  })
  
  # downloading PNG -----
  output$downloadPlotPNG_biplot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("PCA_Biplot",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_biplot, height = input$height_png_biplot, res = input$resolution_PNG_biplot)
      grid.arrange(vals$pcabiplot)
      dev.off()}
  )
  # ======= PCA component
  output$pcacomp <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    pcacomp = fviz_eig(pca)
    vals$pcacomp = pcacomp
  })
  
  # downloading PNG -----
  output$downloadPlotPNG_scree <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("PCA_Component_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_scree, height = input$height_png_scree, res = input$resolution_PNG_scree)
      grid.arrange(vals$pcacomp)
      dev.off()}
  )
  # ======= PCA variable importance
  output$variableimp <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
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
  
  # downloading PNG -----
  output$downloadPlotPNG_genepca <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("PCA_Importance_Genes_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_genepca, height = input$height_png_genepca, res = input$resolution_PNG_genepca)
      grid.arrange(vals$pcavar)
      dev.off()}
  )
  
  output$thetablestats <- DT::renderDataTable({
    DT::datatable(Datastats(), options = list(scrollX = TRUE))
  },
  server = TRUE)
  # =========================================================== Clustring
  
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
  # downloading PNG -----
  output$downloadPlotPNG_heatmap <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Heatmap_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_heatmap, height = input$height_png_heatmap, res = input$resolution_PNG_heatmap)
      grid.arrange(vals$heatmapplot)
      dev.off()}
  )
  
  output$missmap <- renderPlot({
    missmap(Datastatsclust())
  })
  
  ##========== Kmeans
  output$kmeanscluster <- renderPlotly({
    res.km <- kmeans(scale(Datastatsclust()), input$kmeansbins, nstart = 25)
    # Clustering K-means montrant le groupe de chaque individu
    kmeanscluster = fviz_cluster(res.km, data = Datastatsclust(),
                                 # palette = c("#2E9FDF", "#00AFBB", "#E7B800", '#E87722'), 
                                 geom = "point",
                                 ellipse.type = "convex", 
                                 ggtheme = theme_bw()
    )
    vals$kmeanscluster = kmeanscluster
  })
  # downloading PNG -----
  output$downloadPlotPNG_kmeanscluster <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Kmeans_clusters_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_kmeanscluster, height = input$height_png_kmeanscluster, res = input$resolution_PNG_kmeanscluster)
      grid.arrange(vals$kmeanscluster)
      dev.off()}
  )
  
  output$kmeansclusterannot <- renderPlotly({
    res.km <- kmeans(scale(Datastatsclust()), input$kmeansbins, nstart = 25)
    # Réduction de dimension en utilisant l'ACP
    res.pca <- prcomp(Datastatsclust(),  scale = TRUE)
    # Coordonnées des individus
    ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
    # Ajouter les clusters obtenus à l'aide de l'algorithme k-means
    ind.coord$cluster <- factor(res.km$cluster)
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
  
  # downloading PNG -----
  output$downloadPlotPNG_kmeansclusterannot <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Kmeans_clustrs_annotated_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_kmeansclusterannot, height = input$height_png_kmeansclusterannot, res = input$resolution_PNG_kmeansclusterannot)
      grid.arrange(vals$kmeansclusterannot)
      dev.off()}
  )
  
  output$kmeanselbow <- renderPlotly({
    kmeanselbow = fviz_nbclust(Datastatsclust(), kmeans, method = "wss") +
      geom_vline(xintercept = input$kmeansbins, linetype = 2)+
      labs(subtitle = "Elbow method") 
    vals$kmeanselbow = kmeanselbow
  })
  
  # downloading PNG -----
  output$downloadPlotPNG_kmeanselbow <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", Sys.Date())
      paste("Kmeans_Elbow_method_",input$title, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = input$width_png_kmeanselbow, height = input$height_png_kmeanselbow, res = input$resolution_PNG_kmeanselbow)
      grid.arrange(vals$kmeanselbow)
      dev.off()}
  )
  
  #++++++++++++
  output$thetablestatsclust <- DT::renderDataTable({
    DT::datatable(Datastatsclust(), options = list(scrollX = TRUE))
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
  ## =======================================================================================. IGV =========================================================================================================#
  # This are for the server close
})

# =======================================================================================. App. ============================================================================================================#

shinyApp(ui = ui, server = server)

# =======================================================================================. End. ============================================================================================================#