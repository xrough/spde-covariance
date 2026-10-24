# Explicit-rank parity only. This script needs base R, not the covcomp package.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: Rscript scripts/check_author_completion.R FIXTURE_DIRECTORY")
root <- args[[1L]]
revision <- "0d332d99583315b60e17b6fd83b6c2e0b74e8da0"
source_url <- paste0("https://raw.githubusercontent.com/kgwstat/covcomp/", revision, "/R/covcomp.R")
author_file <- tempfile(fileext = ".R")
download.file(source_url, author_file, mode = "wb", quiet = TRUE)
author <- new.env(parent = baseenv())
sys.source(author_file, envir = author)
manifest <- read.csv(file.path(root, "manifest.csv"), stringsAsFactors = FALSE)
if (nrow(manifest) != 8L) stop("expected six paper controls and two PSD controls")
read_matrix <- function(name, suffix) {
  as.matrix(read.csv(file.path(root, paste0(name, "-", suffix, ".csv")), header = FALSE))
}
errors <- numeric(nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  input <- read_matrix(row$name, "input")
  mask <- read_matrix(row$name, "mask")
  expected <- read_matrix(row$name, "expected")
  actual <- author$Compln(input, mask, rep(row$rank, row$joins))
  errors[i] <- max(abs(actual - expected))
  # The finite-rank overlap inversion is ill-conditioned; this is a numerical
  # cross-language tolerance, not an interval or statistical error certificate.
  tolerance <- 1e-8 + 1e-7 * max(abs(expected))
  if (!all(is.finite(actual)) || errors[i] > tolerance) {
    stop(sprintf("%s: max error %.12g exceeds %.12g", row$name, errors[i], tolerance))
  }
  if (max(abs(actual[mask == 1] - input[mask == 1])) != 0) {
    stop(paste(row$name, "changed an observed entry"))
  }
  cat(sprintf("PASS %s rank=%d max_abs_difference=%.12g\n", row$name, row$rank, errors[i]))
}
write.csv(data.frame(name = manifest$name, max_abs_difference = errors),
          file.path(root, "r-parity-results.csv"), row.names = FALSE)
cat(sprintf("Pinned author revision %s: all %d explicit-rank fixtures passed.\n", revision, nrow(manifest)))
