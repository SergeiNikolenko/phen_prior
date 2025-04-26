library(ClinPrior)

args <- commandArgs(trailingOnly = TRUE)
HPOPatient <- unlist(strsplit(args[1], ","))

if (length(HPOPatient) < 2) {
    HPOPatient <- c(HPOPatient, "HP:0000118")
}

Y <- proteinScore(HPOPatient)
ClinPriorGeneScore <- MatrixPropagation(Y, alpha = 0.2)

colnames(ClinPriorGeneScore) <- make.names(colnames(ClinPriorGeneScore), unique = TRUE)
output_file <- paste0("/mnt/", args[2], "_clinprior.csv")
write.csv(ClinPriorGeneScore, output_file, row.names = FALSE)
