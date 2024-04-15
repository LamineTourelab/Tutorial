
# Define downloading PNG function -----
func_save_png <- function(titlepng = " ", img = vals$pcafig, width = input$width_png, height = input$height_png, res = input$resolution_PNG){
  PNG <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", format(Sys.time(), "%a_%b_%d_%Y_%X"))
      paste(titlepng, gsub("/", "-", x), ".png", sep = "")
    },
    content = function(file) {
      
      png(file, width = width, height = height, res = res)
      grid.arrange(img)
      dev.off()}
  )
  return(PNG)
}

# Define downloading PDF function -----

func_save_pdf <- function(titlepdf = " ", img = vals$pcafig, width = input$width, height = input$height){
  PDF <- downloadHandler(
    filename = function() {
      x <- gsub(":", ".", format(Sys.time(), "%a_%b_%d_%Y_%X"))
      paste(titlepdf, gsub("/", "-", x), ".pdf", sep = "")
    },
    content = function(file) {
      pdf(file, width = width, height = height, onefile = FALSE) # open the pdf device
      grid.arrange(img)
      dev.off()},
    
    contentType = "application/pdf"
    
  )
  return(PDF)
}

