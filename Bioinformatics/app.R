library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)
library(ggrepel)
library(plotly)
library(tidyr)
library(tidyverse)
library(dplyr)
library(factoextra) # for pca visualization 
library(FactoMineR)  # PCA
library(enrichR)
library(gplots) # Use heatmap.2
library(corrplot)
library(mixOmics) # for the breast cancer dataset
library(Amelia) # for missing values visualization
library(igvShiny)
library(shinymanager)
library(hypeR)

## ==================================================================== Datasets ============================================================================================##
data(breast.TCGA) # from the mixomics package.
mRna = data.frame(breast.TCGA$data.train$mrna)
mRna$subtype = breast.TCGA$data.train$subtype
Transcriptomics_data <- readr::read_csv("https://raw.githubusercontent.com/LamineTourelab/Tutorial/main/Data%20Visualization/Shiny%20App%20in%20genomics/Data/Transcriptomics%20data.csv")
stock.genomes <- sort(get_css_genomes())
# ==================================================================== Options ===================================================================================================#
options(shiny.maxRequestSize = 50*1024^2) 
# ======================================================================  Ui. =================================================================================================##


dashHeader = dashboardHeader(title ="My Dashboard",
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
 # sidebarSearchForm(label = "Enter a number", "searchText", "searchButton"),
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
      icon = icon('chart-line', style = "color:#E87722")),
    
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
      text = 'File Explore',
      tabName = 'FileExplore',
      icon = icon('file-text', style = "color:#E87722")),
    
    menuItem(text = 'Javascript',
             tabName = 'JS',
             icon = icon('code', style = "color:#E87722"))
  )
)

dashbody <- dashboardBody(
  shinyjs::useShinyjs(),
  # ================================================================================  Graph
  tabItems(
    tabItem(tabName = 'hometab',
            h1('Home  page!'),
            img(src = "inem.jpeg", height = 72, width = 72),
            p('This is a home page for dashboard, it will be developped later.')
    ),
    tabItem(tabName = 'Graphstab', 
          
            fluidRow(
              sidebarPanel(width = 2, height = 1170,
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
              tabBox(width = 10,
                     tabPanel(title='Histogram & Boxplot',
                              #Placeholder for plot
                              fluidRow(
                              column(6, plotlyOutput(outputId='Histplot')),
                              column(6, plotlyOutput(outputId='boxplot')),
                              )
                        ),
                     tabPanel(title='Linear & Density',
                              fluidRow(
                                column(6, plotlyOutput(outputId='linearplot')),
                                column(6, plotlyOutput(outputId='density'))
                              )
                          
                     ),
                     tabPanel(title='Data Table',
                              DT::dataTableOutput(outputId = 'thetable'),
                             # verbatimTextOutput("summarythetable")
                     )
                     
              )
            )
    ),
    
    # # ================================================================================  Statistical Analysis.
    tabItem(tabName = 'Statistics',
            fluidRow(
              sidebarPanel(width = 2, height = 1170,
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
              tabBox( width = 10,
                      tabPanel(title='PCA',
                               #Placeholder for plot
                               fluidRow(
                                 column(6, plotlyOutput(outputId='pca')),
                                 column(6, plotlyOutput(outputId='pcabiplot'))
                               ),
                               fluidRow(
                                 column(6, plotlyOutput(outputId='pcacomp')),
                                 column(6, plotlyOutput(outputId='variableimp'))
                               )
                      ),
                      tabPanel(title='Test',
                               #Placeholder for plot
                               plotlyOutput(outputId='stattest'),
                      ),
                      tabPanel(title='Heatmap',
                               #Placeholder for plot
                               plotOutput(outputId = 'missmap'),
                               plotOutput(outputId='heatmap'),
                      ),
                      tabPanel(title='Correlation plot',
                               #Placeholder for plot
                               plotlyOutput(outputId='Corrplot')
                      ),
                      tabPanel(title='Data Table',
                               DT::dataTableOutput(outputId = 'thetablestats')
                               
                      )
                  ) # tabBox
              )
      
    ), # tabItem for statistics
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
                               plotlyOutput(outputId = 'Volcanoplot')
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
      shinyUI(fluidPage(igvShinyOutput('igvShiny'), width = 10))
    ),
    # ================================================================================  Enrichment
    tabItem(
      tabName ='enrich',
      fluidPage(
        sidebarLayout(
          sidebarPanel(
            textAreaInput("genes", "Enter genes names (separed by a ',') :"),
            actionButton("submit", "Submit"),
            p(style="text-align: justify;",
              "Here you can choose a database to see the results in data table format and/or plot."),
            hr(style="border-color: blue;"),
            selectInput("databaseenrich", "Choose a database:", choices = dbs)
          ),
          mainPanel(tabsetPanel(
            tabPanel(title = 'Enrichment',
                     #  verbatimTextOutput("results"),
                     DT::dataTableOutput(outputId = 'thetableenrich'),
            ),
            tabPanel(title = 'Plot',
                     plotlyOutput(outputId='gsea'),
            )
          )
          ) # mainPanel
          
        )
      ) # fluidPage
      
    ), #tabItem
    # # ================================================================================  Exploring file
    tabItem(
      tabName ='FileExplore',
      fileInput(
        inputId='fileupload',
        label = 'Please upload a file',
        multiple = FALSE,
        accept = c('csv', 'text'),
        buttonLabel = 'Upload',
        placeholder = 'Waiting on a file'
      ),
      fluidRow(
        box(width = 3,
            checkboxGroupInput(
              inputId='Colselection',
              label = 'Waiting for data',
              choices = NULL
            )
        ),
        box(width = 9,
            DT::dataTableOutput(outputId = 'FileTable')
            
        )
      )
    ), #tabItem for DEA
    # ================================================================================  JS
    tabItem(
      tabName = 'JS',
      fluidRow(
        box(width = 4, height = 200,
            h1(id='Leftext', 'Left')
            
        ),
        box(width = 4, height = 200,
            actionButton(
              inputId='SwapButton',
              label = 'Swap'
            )
        ),
        box(width = 4, height = 200,
            shinyjs::hidden(h1(id='Righttext', 'Right'))
            
        )
        
      ),
      fluidRow(
        box(
          width = 4,
          textInput(
            inputId='NumGuess',
            label = NULL,
            placeholder = 'Guess a number'
          )
        ),
        box(
          width = 4,
          shinyjs::hidden(
            h3(id='LuckyNumber', 'you guess the correct number')
          )
        ),
        box(
          width = 4,
          actionButton(
            inputId='AlertButton',
            label = 'Ring',
            class='btn-primary'
          )
        )
      )
    )
  )
)


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
      label = 'Please choose a color',
      choices = c('NULL',names(Datagraph())),
      selected = 'subtype'
    )
  })
  
  output$Histplot <- renderPlotly({
    Hist <- ggplot(Datagraph(), aes_string(x=input$Vartoplot, fill=input$VarColor)) +  geom_histogram(bins = input$histbins)
    Hist %>% 
      ggplotly(tooltip = 'all')
  })
  
  output$density <- renderPlotly({
    Dens <- ggplot(Datagraph(), aes_string(x=input$Vartoplot1, fill=input$VarColor1)) +  geom_density(fill='grey50')
    Dens %>% 
      ggplotly(tooltip = 'all') %>%
      layout(dragmode = "select")
  })
  
  output$boxplot <- renderPlotly({
    Boxp <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_boxplot() 
    Boxp %>% 
      ggplotly(tooltip = 'all') 
  })
  
  output$linearplot <- renderPlotly({
    
    if(!input$Vartofill == 'NULL'){
      Corrp <-  ggplot(Datagraph(), aes_string(input$VarColor1, input$Vartoplot1 , fill=input$Vartofill)) + geom_point(position = "jitter") 
      Corrp %>% 
        ggplotly(tooltip = 'all')
    }else{
      Corrp <-  ggplot(Datagraph(), aes_string(input$VarColor1, input$Vartoplot1)) + geom_point(position = "jitter") 
      Corrp %>% 
        ggplotly(tooltip = 'all')
    }
  })
  
  
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
                      ' which represents ' ,round(totalNO/total*100,2),'% of the data',sep='')
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
    volcano_gplot %>% 
      ggplotly(tooltip = 'all') 
    })
  
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
    
    dbs <- c("GO_Molecular_Function_2015", "GO_Cellular_Component_2015", "GO_Biological_Process_2015",
             "Reactome_2015", "Reactome_2016", "OMIM_Disease", "MSigDB_Oncogenic_Signatures", "KEGG_2015",
             "KEGG_2016", "GO_Biological_Process_2018", "Human_Phenotype_Ontology", "Cancer_Cell_Line_Encyclopedia",
             "RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO")
    
    # Exécuter l'analyse d'enrichissement de gènes avec enrichR
    enrichment_result <- enrichr(user_genes, dbs)
    
    # Afficher les résultats dans la sortie
    output$results <- renderPrint({
      head(enrichment_result[[input$databaseenrich]])
    })
    
    output$thetableenrich <- DT::renderDataTable({
      DT::datatable(enrichment_result[[input$databaseenrich]], rownames = TRUE, options = list(scrollX = TRUE))
    },
    server = TRUE)
    
    output$gsea <- renderPlotly({
      
      plotEnrich(enrichment_result[[input$databaseenrich]], showTerms = 20, numChar = 50,
                 y = "Count", orderBy = "P.value", title = "Enrichment analysis",
                 xlab = "Enriched pathways")
    })
  })
  
    ## =========================================================================.  Statistical Panel results.  =============================================================================== #
  
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
  ##========== plot
  
  output$heatmap <- renderPlot({
    Datastats <- as.matrix(Datastats())
    heatmap.2(Datastats, col = greenred(256), scale="column", margins=c(5,10), density="density", xlab = "Gene Names", 
              ylab= "Samples ID", main = " ",breaks=seq(-1.5,1.5,length.out=257))
  })
  
  output$missmap <- renderPlot({
    missmap(Datastats())
  })
  
  output$pca <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    fviz_pca_ind(pca, fill.ind = metadata[,1],  geom.ind = "point", 
                 pointshape=21,addEllipses = F,pointsize=4 )
  })
  
  output$pcabiplot <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    fviz_pca_biplot(pca, fill.ind = metadata[,1], geom.ind = "point", 
                  pointshape=10,addEllipses = T,pointsize=2)
  })
  
  output$pcacomp <- renderPlotly({
    Datastats <- as.matrix(Datastats())
    metadata = Metadastats() %>% dplyr::select(input$Vartoplotstats)
    pca = FactoMineR::PCA(Datastats, scale.unit=T, graph=F)
    
    fviz_eig(pca)
  })

  
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
   pcavar %>% 
     ggplotly(tooltip = 'all') 
  })
  
  output$thetablestats <- DT::renderDataTable({
    DT::datatable(Datastats(), options = list(scrollX = TRUE))
  },
  server = TRUE)
      ## ==================================================================. Exploring file.   =================================================================================#
  
  
  theData <- reactive({
    if(is.null(input$fileupload)){
      return(NULL)
    }
    readr::read_csv(input$fileupload$datapath)
  })
  
  observe({
    updateCheckboxGroupInput(
      inputId = 'Colselection',
      session = session,
      label = 'Please choose colunms to disaplay',
      choices = names(theData()),
      selected = names(theData())
    )
  })
  
  output$FileTable <- DT::renderDataTable({
    DT::datatable(theData()[,input$Colselection, drop=FALSE], rownames = TRUE,options = list(scrollX = TRUE))
  },
  server = TRUE)
  observeEvent(input$SwapButton, {
    shinyjs::toggle(id='Leftext')
    shinyjs::toggle(id='Righttext')
  })
  
  observeEvent(input$AlertButton, {
    shinyjs::alert('This is a alert.')
  })
  
  observe({
    shinyjs::toggle(id='LuckyNumber', condition = input$NumGuess==5)
    shinyjs::toggleState(id='AlertButton', condition = input$NumGuess==1)
  })
  
  #output$Report <- downloadHandler(
  #  filename = 'Report.html',
   # content = function(file){
   #   tempfile = file.path(tempdir(), 'report.rmd')
 #     file.copy('report.Rmd', to= tempfile, overwrite = TRUE)
   #   
    #  chosen <- 
   # }
 # )
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