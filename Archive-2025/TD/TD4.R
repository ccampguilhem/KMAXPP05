f <- function(x){ exp(-x) * sin(2*pi*x) }
curve(f, from = 0, to = 1, lwd = 2, col = "black")

y <- c(1/4, 3/4)
fx <- f(y)

cx <- function(t){
  c(min(t, 1/4), min(t, 3/4))
}

Sigma_inv <- matrix(c(6, -2, -2, 2), nrow = 2)

fpred <- function(x, y){
  as.numeric(cx(x) %*% Sigma_inv %*% f(y))
}

fpred(1:10, y)

curve(expr = fpred(x, y = c(1/4, 3/4)),
      from = 0, to = 1,
      add = TRUE, col = "blue", lwd = 2)

points(y, fx, pch = 19, col = "red")
legend("topright", legend = c("Vraie fonction", "Prédiction", "Points observés"),
       col = c("black", "blue", "red"), lwd = c(2,2,NA), pch = c(NA, NA, 19))
