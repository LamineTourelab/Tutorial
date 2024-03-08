library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)


data(diamonds, package = 'ggplot2')


# Ui

dashHeader = dashboardHeader(title = 'Simple Shinydashboard')
dashsidebar = dashboardSidebar(
  sidebarMenu(
    menuItem(
      text = 'home', 
      tabName = 'hometab',
      icon = icon('dashboard', style = "color:#E87722")),
    
    menuItem(
      text = 'Graphs',
      tabName = 'Graphstab',
      icon = icon('chart-column', style = "color:#E87722")),
    
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
                  selectInput("dataset", "Choose a dataset:", choices = c("test-data", "own")),
                  fileInput(inputId = 'file1', 'ID, logFC, Pvalue',
                            accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                  # Placeholder for input selection
                  selectInput('Vartoplot', 
                              label = 'choose a variable',
                              choices = c('carat', 'depth', 'table', 'price'),
                              selected = 'price'),
                  # Choose number of bins
                  sliderInput(inputId='histbins',
                              label = 'please select a number of bins',
                              min = 5, max = 50, value = 30)
              ),
              tabBox(width = 10,
                     tabPanel(title='Histogram',
                              #Placeholder for plot
                              plotOutput(outputId='Histplot'),
                              h4("Exporting the ggVolcanoR plot"),
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
                     tabPanel(title='Volvano plot',
                              plotOutput(outputId = 'Volcanoplot'),
                              
                              ),
                     
                     tabPanel(title='Density',
                              plotOutput(outputId='density')
                     ),
                     tabPanel(title='Table',
                              DT::dataTableOutput(outputId = 'thetable')
                              
                     )
                     
              )
            )
    ),
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

# Server

server <- shinyServer(function(input, output, session)
{
  output$Histplot <- renderPlot({
    ggplot(diamonds, aes_string(x=input$Vartoplot)) +  geom_histogram(bins = input$histbins)
  })
  
  Datav <- reactive({
    if(is.null(input$file1)){
      return(NULL)
    }
    readr::read_csv(input$file1$datapath)
  })
  
  output$Volcanoplot <- renderPlot({
    # add a column of NAs
    Datav = data.frame(Datav())
    Datav$diffexpressed <- "NO"
    # if log2Foldchange > 0.6 and pvalue < 0.05, set as "UP" 
    Datav$diffexpressed[Datav$logFC > 0.6 & Datav$Pvalue < 0.05] <- "UP"
    # if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"
    Datav$diffexpressed[Datav$logFC < -0.6 & Datav$Pvalue < 0.05] <- "DOWN"
    
    # Now write down the name of genes beside the points...
    # Create a new column "delabel" to de, that will contain the name of genes differentially expressed (NA in case they are not)
    Datav$delabel <- NA
    Datav$delabel[Datav$diffexpressed != "NO"] <- Datav$ID[Datav$diffexpressed != "NO"]
    
    volcano_gplot <- ggplot(data=Datav, aes(x=logFC, y=-log10(Pvalue), col=diffexpressed, label=delabel)) + 
      geom_point() + 
      theme_minimal() +
      geom_text_repel() +
      scale_color_manual(values=c("blue", "black", "red")) +
      # Add vertical lines for log2FoldChange thresholds, and one horizontal line for the p-value threshold 
      geom_vline(xintercept=c(-0.6, 0.6), col="red") +  
      geom_hline(yintercept=-log10(0.05), col="red")
    volcano_gplot
    })
  
  output$density <- renderPlot({
    ggplot(diamonds, aes_string(x=input$Vartoplot)) +  geom_density(fill='grey50')
  })
  
  output$thetable <- DT::renderDataTable({
    DT::datatable(diamonds, rownames = FALSE)
  },
  server = TRUE)
  
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

# App
shinyApp(ui = ui, server = server)
