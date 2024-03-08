library(shiny)
library(shinydashboard)
library(shinyjs)

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
            fluidRow(
              tabBox(
                box( 
                  sidebarPanel(
                    id = "tPanel",style = "overflow-y:scroll; max-height: 900px; position:relative;",
                    tags$style(type="text/css", "body {padding-top: 70px; padding-left: 10px;}"),
                    tags$head(tags$style(HTML(".shiny-notification {position:fixed;top: 50%;left: 30%;right: 30%;}"))),
                    tags$head(tags$style(HTML('.progress-bar {background-color: blue;}'))),
                    
                    #selectInput("user.defined","Types of preset parameters",choices = style.volcano.type),
                    selectInput("dataset", "Choose a dataset:", choices = c("test-data", "own")),
                    fileInput('file1', 'ID, logFC, Pvalue',
                              accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
                    fluidRow(
                      column(6,radioButtons('sep', 'Separator', c( Tab='\t', Comma=','), ',')),
                      column(6,radioButtons('quote', 'Quote', c(None='', 'Double Quote'='"', 'Single Quote'="'"), '"'))
                    )
                  ),
                ),
                box(
                  collapsible = TRUE,
                  title = 'controls',
                  status = 'success', solidHeader = TRUE,
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
              
              ),
              tabBox(
                     tabPanel(title='Histogram',
                       #Placeholder for plot
                       plotOutput(outputId='Histplot')
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
  
dashboardPage(
  header = dashHeader ,
  sidebar = dashsidebar,
  body = dashbody,
  title = 'Example dashboard',
  skin = 'blue'
)