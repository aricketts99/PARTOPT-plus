library(reticulate)
library(sf)
library(spdep)

## ------------------------------------------------------------
## Set up the Python environment (one-time installation)
## ------------------------------------------------------------

if (!"partopt-env" %in% conda_list()$name) {
  conda_create(
    envname = "env-partopt",
    environment = "environment.yml"
  )
  conda_run2(
    cmd = "python",
    args = c(
      "-m", "pip",
      "install",
      "-e",
      "."
    ),
    envname = "env-partopt"
  )
  
}



use_condaenv("env-partopt", required = TRUE)


## Import Python modules
nx <- import("networkx")
partopt <- import("partopt_plus")
Partition_Searcher <- partopt$Partition_Searcher

## ------------------------------------------------------------
## Example parameters
## ------------------------------------------------------------

n <- 196L
rs <- as.integer(sqrt(n))

d <- 4L
m <- 5L
L <- 5L

rho <- 0.95
mu <- 1000

prior_1 <- 1
prior_2 <- 0

tau <- 0.5
lambda_ <- 0.5

seed <- 12345L

tau_prior_1 <- 0.5
tau_prior_2 <- 0.5

k_0 <- 0.01

## ------------------------------------------------------------
## Generate example R data
## ------------------------------------------------------------

set.seed(seed)

X <- array(
  rnorm(n * m * d, mean = 1, sd = 1),
  dim = c(n, m, d)
)

## Constant intercept column
X[, , 1] <- 1

assignment <- rep(0:3, each = n / 4)

Y <- matrix(
  rnorm(n * m),
  nrow = n,
  ncol = m
)

## ------------------------------------------------------------
## Construct a Queen contiguity graph from an sf grid
## ------------------------------------------------------------

bbox <- st_bbox(c(
  xmin = 0,
  ymin = 0,
  xmax = rs,
  ymax = rs
))

grid <- st_make_grid(
  st_as_sfc(bbox),
  n = c(rs, rs),
  what = "polygons"
)

grid <- st_sf(
  id = seq_along(grid),
  geometry = grid
)

## Queen contiguity neighbours
nb <- poly2nb(grid, queen = TRUE)

## Binary adjacency matrix
W <- nb2mat(
  nb,
  style = "B",
  zero.policy = TRUE
)

## Convert the adjacency matrix to a NetworkX graph.
## reticulate automatically converts the R matrix to a NumPy array.
G <- nx$from_numpy_array(W)

## ------------------------------------------------------------
## Run PARTOPT+
## ------------------------------------------------------------

searcher <- Partition_Searcher$Partition_Searcher(
  actual_assignment = assignment,
  Y = Y,
  X = X,
  W = G,
  L = L,
  Kmeans_initliaise = TRUE,
  mu = mu,
  lambda_ = lambda_,
  k_min = 2L,
  k_max = 4L,
  k_0 = k_0,
  rho = rho,
  alpha = tau_prior_1,
  beta = tau_prior_2,
  prior_alpha = prior_1,
  prior_theta = prior_2,
  z_start = 0L
)

searcher$instantiate_set()

searcher$optimise_PARTOPT(
  zeal = 1L,
  global_m = 1L
)
