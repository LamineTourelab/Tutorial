library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)


data(diamonds, package = 'ggplot2')


# Ui

ui <- navbarPage("App Title", inverse = TRUE,
                 tabPanel("Plot"),
                 navbarMenu("More",
                            tabPanel("Summary"),
                            "----",
                            "Section header",
                            tabPanel("Table")
                 )
)
server  <- function(input, output, session) {
  
}


shinyApp(ui, server)


