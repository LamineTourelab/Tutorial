library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)
library(ggrepel)
library(plotly)
library(tidyr)
library(tidyverse)
library(dplyr)
library(factoextra)
library(FactoMineR)
library(enrichR)
library(ggVolcanoR)
library(gplots)
library(corrplot)
library(mixOmics)

## ==================================================================== Datasets ============================================================================================##

data(breast.TCGA) # from the mixomics package.
mRna = data.frame(breast.TCGA$data.train$mrna)
mRna$subtype = breast.TCGA$data.train$subtype
Proteomics_data <- readr::read_csv("https://raw.githubusercontent.com/LamineTourelab/Tutorial/main/Data%20Visualization/Shiny%20App%20in%20genomics/Data/Proteomics%20data.csv")
# ======================================================================  Ui. =================================================================================================##

dashHeader = dashboardHeader(title = 'BioInfo HUB INEM ShinyApp')
dashsidebar = dashboardSidebar(
  sidebarMenu(
    menuItem(
      text = 'Home', 
      tabName = 'hometab',
      icon = icon('dashboard', style = "color:#E87722")),
    
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
  tabItems(
    tabItem(tabName = 'hometab',
            h1('Landing  page!'),
            p('This is a landing page for dashboard'),
            em('This is a emphasize text')
    ),
    tabItem(tabName = 'Graphstab', 
            h1('Graphs!'),
          #  fluidRow(
          #    box(
           #     width = 12,
           ##     downloadButton(outputId='Report', label = 'Generate report')
            #  )
            #),
            fluidRow(
              box(width = 2, height = 1170,
                  collapsible = TRUE,
                  title = 'Side bar',
                  status = 'primary', solidHeader = TRUE,
                  p("Here you can upload you own data by changing the mode test-data to own"),
                  selectInput("datasetgraph", "Choose a dataset:", choices = c("test-data", "own")),
                  fileInput(inputId = 'filegraph', 'Please upload a matrix file'),
                  p("Here you can choose a variable to plot and for coloring. By default you can select the cancer subtype (the last variable of the color list) for color variable."),
                  # Placeholder for input selection
                  fluidRow(
                    column(6, selectInput(inputId ='Vartoplot', label = 'Waiting for data', choices = NULL)),
                    column(6, selectInput(inputId='VarColor',label = 'Waiting for data Color', choices = NULL))
                  ),
                  # Choose number of bins
                  sliderInput(inputId='histbins',
                              label = 'please select a number of bins',
                              min = 5, max = 50, value = 30)
              ),
              tabBox(width = 10,
                     tabPanel(title='Histogram',
                              #Placeholder for plot
                              fluidRow(
                              column(6, plotlyOutput(outputId='Histplot')),
                              column(6, plotlyOutput(outputId='density'))
                              )
                        ),
                     tabPanel(title='Box Plot',
                              fluidRow(
                                column(6, plotlyOutput(outputId='boxplot')),
                                column(6, plotlyOutput(outputId='corrplot'))
                              )
                              #plotlyOutput(outputId='boxplot'),
                             # plotlyOutput(outputId='corrplot')
                     ),
                     tabPanel(title='Table',
                              DT::dataTableOutput(outputId = 'thetable')
                              
                     )
                     
              )
            )
    ),
    
    # Statistical Analysis.
    tabItem(tabName = 'Statistics',
            fluidRow(
              box(width = 2, height = 1170,
                  collapsible = TRUE,
                  title = 'Side bar',
                  status = 'primary', solidHeader = TRUE,
                  selectInput("datasetstats", "Choose a dataset:", choices = c("test-data", "own")),
                  fileInput(inputId = 'filestats', 'Please upload a matrix file',
                            accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                  # Placeholder for input selection
                  fluidRow(
                    column(6, selectInput(inputId='Vartoplotstats',label = 'Waiting for data',choices = NULL )),
                    column(6, selectInput(inputId='VarColorstats',label = 'Waiting for data Color',choices = NULL ))
                  ),
                  # Choose number of bins
                  sliderInput(inputId='histbins1',
                              label = 'please select a number of bins',
                              min = 5, max = 50, value = 30)
              ),
              tabBox( width = 10,
                      tabPanel(title='PCA',
                               #Placeholder for plot
                               plotlyOutput(outputId='pca'),
                               plotOutput(outputId='pcacomp'),
                               plotlyOutput(outputId='variableimp')
                      ),
                      tabPanel(title='Test',
                               #Placeholder for plot
                               plotlyOutput(outputId='stattest'),
                      ),
                      tabPanel(title='Heatmap',
                               #Placeholder for plot
                               plotOutput(outputId='heatmap'),
                      ),
                      tabPanel(title='Correlation plot',
                               #Placeholder for plot
                               plotlyOutput(outputId='Corrplot')
                      )
                  )
              )
      
    ),
    # Differential expression Analysis.
    tabItem(tabName = 'diffexp',
            fluidRow(
              box(width = 2, height = 1170,
                  collapsible = TRUE,
                  title = 'Side bar',
                  status = 'primary', solidHeader = TRUE,
                  selectInput("datasetdiff", "Choose a dataset:", choices = c("test-data", "own")),
                  p("Here you can upload you own data by changing the mode test-data to own. The data should be in the format :ID, logFC, Pvalue."),
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
                               #Placeholder for plot
                               textInput(inputId = 'textinputdiff', label = 'Paste a gene list'),
                               plotlyOutput(outputId='gsea'),
                      )
              )
            )
            
    ),
    # Exploring file
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
    ),
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
  
  test.data.graph <- reactive ({ mRna
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
      
    )
  })
  
  output$Histplot <- renderPlotly({
    Hist <- ggplot(Datagraph(), aes_string(x=input$Vartoplot, fill=input$VarColor)) +  geom_histogram(bins = input$histbins)
    Hist %>% 
      ggplotly(tooltip = 'all')
  })
  
  output$density <- renderPlotly({
    Dens <- ggplot(Datagraph(), aes_string(x=input$Vartoplot, fill=input$VarColor)) +  geom_density(fill='grey50')
    Dens %>% 
      ggplotly(tooltip = 'all') %>%
      layout(dragmode = "select")
  })
  
  output$boxplot <- renderPlotly({
    Boxp <- ggplot(Datagraph(), aes_string(input$VarColor, input$Vartoplot, fill=input$VarColor)) +  geom_boxplot() + facet_grid(. ~ input$VarColor, scales = "free", space = "free")
    Boxp %>% 
      ggplotly(tooltip = 'all') 
  })
  
  output$corrplot <- renderPlotly({
    Corrp <-  ggplot(Datagraph()) + geom_point(aes_string(input$VarColor, input$Vartoplot , fill=input$VarColor), position = "jitter")
    Corrp %>% 
      ggplotly(tooltip = 'all') 
  })
  
  
  output$thetable <- DT::renderDataTable({
    DT::datatable(Datagraph(), rownames = FALSE)
  },
  server = TRUE)
  
      ## =========================================================================.  Differntial Panel results.  =============================================================================== #
  
  Datadiff <- reactive({switch(input$datasetdiff,"test-data" = test.data(),"own" = own.data())})
  
  test.data <- reactive({
   # dataframe = read.csv(system.file("extdata","Proteomics data.csv",package = "ggVolcanoR"),header = T)
    Proteomics_data
  })
  own.data <- reactive({
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
    dat$diffexpressed <- "NO"
    dat$diffexpressed[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$diffexpressed[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$diffexpressed != "NO"] <- dat$ID[dat$diffexpressed != "NO"]
    dat <- dat[order(dat$Pvalue),]
    dat$logP <- -log10(dat$Pvalue)
    total <- as.numeric(dim(dat)[1])
    totalDown <- as.numeric(dim(dat[dat$diffexpressed=='DOWN',])[1])
    totalNO <- as.numeric(dim(dat[dat$diffexpressed=='NO',])[1])
    totalUP <- as.numeric(dim(dat[dat$diffexpressed=='UP',])[1])
    
    cat('\n There are a total of ', total, ' where '  , totalDown, ' are dowregulated ', totalUP, ' are upregulated and ', totalNO, ' are none. ', 
                      ' which represents ' ,round(totalNO/total*100,2),'% of the data',sep='')
  })
    
  output$Volcanoplot <- renderPlotly({
    
    Datadiff = data.frame(Datadiff())
    # add a column of NAs
    Datadiff$diffexpressed <- "NO"
    # if log2Foldchange > 0.6 and pvalue < 0.05, set as "UP" 
    Datadiff$diffexpressed[Datadiff$logFC > 0.6 & Datadiff$Pvalue < 0.05] <- "UP"
    # if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"
    Datadiff$diffexpressed[Datadiff$logFC < -0.6 & Datadiff$Pvalue < 0.05] <- "DOWN"
    
    # Now write down the name of genes beside the points...
    # Create a new column "gene_name" to de, that will contain the name of genes differentially expressed (NA in case they are not)
    Datadiff$gene_name <- NA
    Datadiff$gene_name[Datadiff$diffexpressed != "NO"] <- Datadiff$ID[Datadiff$diffexpressed != "NO"]
    
    volcano_gplot <- ggplot(data=Datadiff, aes(x=logFC, y=-log10(Pvalue), col=diffexpressed, label=gene_name)) + 
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
    dat$diffexpressed <- "NO"
    dat$diffexpressed[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$diffexpressed[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$diffexpressed != "NO"] <- dat$ID[dat$diffexpressed != "NO"]
    
    dat <- DT::datatable(dat)
    dat
  })
  
  output$summarytablecount <- DT::renderDataTable({
    dat <- data.frame(Datadiff())
    dat$diffexpressed <- "NO"
    dat$diffexpressed[dat$logFC > 0.6 & dat$Pvalue < 0.05] <- "UP"
    dat$diffexpressed[dat$logFC < -0.6 & dat$Pvalue < 0.05] <- "DOWN"
    dat$gene_name <- NA
    dat$gene_name[dat$diffexpressed != "NO"] <- dat$ID[dat$diffexpressed != "NO"]
    dat <- dat %>% dplyr::count(diffexpressed)
    DT::datatable(dat)
  })
  
  ##### Enrichment
  
  # output$textinputdiff <- renderPrint({
  #   input$textinputdiff
  # })
  # 
  # output$gsea <- renderPlotly({
  #   listEnrichrSites()
  #   setEnrichrSite("Enrichr")
  #   websiteLive <- TRUE
  #   
  #   dbs <- listEnrichrDbs()
  #   dbs <- c("GO_Molecular_Function_2015", "GO_Cellular_Component_2015", "GO_Biological_Process_2015", 
  #            "Reactome_2015", "Reactome_2016", "OMIM_Disease", "MSigDB_Oncogenic_Signatures", "KEGG_2015", 
  #            "KEGG_2016", "GO_Biological_Process_2018", "Human_Phenotype_Ontology", "Cancer_Cell_Line_Encyclopedia",
  #            "RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO")
  #   if (websiteLive) {
  #     enriched <- enrichr(c("SFRP1", "RELN", "FLT1", "GPC3", "APOBEC3B", "CD47", "NTRK2", "TLR8", 
  #                           "FGF10", "E2F1", "HBEGF", "SLC19A3", "DUSP6", "FOS", "GNG5"), input$Vardatabasediff)
  #   }
  #   
  #   plotEnrich(enriched[[input$Vartoplotdiff]], showTerms = 20, numChar = 50,
  #              y = "Count", orderBy = "P.value", title = "Enrichment analysis with selected Modules using KEGG", 
  #              xlab = "Enriched pathways")
  # })
  
    ## =========================================================================.  Statistical Panel results.  =============================================================================== #
  
  Datastats <- reactive({switch(input$datasetstats,"test-data" = test.data.stats(),"own" = own.data.stats())})
  
  test.data.stats <- reactive ({ 
    mRna = data.frame(breast.TCGA$data.train$mrna)
  })
  
  own.data.stats <- reactive({
    if(is.null(input$filestats)){
      return(NULL)
    }
    dataframe = readr::read_csv(input$filestats$datapath)
    
  })
  
  output$heatmap <- renderPlot({
    Datastats <- as.matrix(Datastats())
    heatmap.2(Datastats, col = greenred(256), scale="column", margins=c(5,10), density="density", xlab = "Gene Names", 
              ylab= "Samples ID", main = " ",breaks=seq(-1.5,1.5,length.out=257))
  })
  
  
  
  output$pca <- renderPlotly({
    pca =PCA(data.frame(Datastats()), scale.unit=T, graph=F)
    
    fviz_pca_ind(pca, fill.ind = breast.TCGA$data.train$subtype , geom.ind = "point", 
                 pointshape=21,addEllipses = F,pointsize=4 )
  })
  
  output$pcacomp <- renderPlot({
  
    fviz_eig(pca)
  })
  
  output$variableimp <- renderPlotly({
    loadings <- data.frame(pca$var$coord[1:10,1:5])
    loadings$Symbol <- row.names(loadings)
    loadings <- gather(loadings, 'Component', 'Weight',
                       -Symbol)
   pcavar <- ggplot(loadings, aes(x=Symbol, y=Weight)) +
      geom_bar(stat='identity', aes(colour = factor(sign(Weight)))) +
      facet_grid(Component ~
                   ., scales='free_y')
   pcavar %>% 
     ggplotly(tooltip = 'all') 
  })

  
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
    DT::datatable(theData()[,input$Colselection, drop=FALSE], rownames = FALSE)
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
  
})

# =======================================================================================. App. ============================================================================================================#

shinyApp(ui = ui, server = server)

# =======================================================================================. End. ============================================================================================================#