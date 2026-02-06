library(tidyverse)

# Exo 1

xtx <- matrix(c(30, 20, 0, 20, 20, 0, 0, 0, 10), nrow = 3)
xty <- matrix(c(15, 20, 10), nrow = 3)
yty = 59.5

xtx1 <- solve(xtx)
beta_hat <- xtx1 %*% xty

# Exo 2
n <- 50
p <- 3

xtx <- matrix(c(50, 0, 0, 0, 0, 20, 15, 4, 0, 15, 30, 10, 0, 4, 10, 40), nrow = 4)
xty <- matrix(c(100, 50, 40, 80), nrow = 4)
yty = 640

xtx1 <- solve(xtx)
beta_hat <- xtx1 %*% xty

carre_residu = yty - t(beta_hat) %*% xty
sigma2_hat <- carre_residu/(n-p)

beta_hat[2] - qt(0.975, n-p)*sqrt(sigma2_hat * xtx1[2,2])
beta_hat[2] + qt(0.975, n-p)*sqrt(sigma2_hat * xtx1[2,2])

(n-p)*sigma2_hat/66
(n-p)*sigma2_hat/29

SSR <- t(beta_hat)%*%xty - n* (xty[1]/n)
SSE <- yty - t(beta_hat)%*%xty

fisher_stat <- (SSR/p)*(SSE/(n-p))

# Exo 3

annee <- 1992:2001
revenu <- c(8000,9000,9500,9500,9800,11000,12000,13000,15000,16000)
conso <- c(7389.99,8169.65,8831.71,8652.84,8788.08,9616.21,10593.45,11186.11,12758.09,13869.62)
df <- data.frame(annee, revenu, conso)

#1)
df %>% ggplot(aes(x = revenu, y = conso)) +
  geom_point()

#2)
beta_hat <- cov(conso, revenu)/var(revenu)
alpha_hat <- mean(conso) - beta_hat*mean(revenu)

#3)
conso_hat <- alpha_hat + beta_hat*revenu
conso_hat

#4)
residu <- conso - conso_hat
mean(residu)

#5)
1/(length(residu)-1) * sum(residu^2)
var(residu)
sd(residu)

#6)
qt(0.975, 8)
beta_hat - qt(0.975, 8) * sd(residu) /sqrt(8 * var(revenu))
beta_hat + qt(0.975, 8) * sd(residu) /sqrt(8 * var(revenu))

#7)
r_square <- 1 - sum(residu^2)/sum((conso - mean(conso))^2)

stat_fisher <- 8*r_square/(1 - r_square)
qf(0.95, 1, 8)



#8)
r2002 <- 16800
r2003 <- 17000

c2002 <- alpha_hat + beta_hat * r2002
c2003 <- alpha_hat + beta_hat * r2003

c2002 - qt(0.975, 8) * sd(residu) * sqrt(1+ 1/10 + (r2002 - mean(revenu))^2/((9)*var(revenu)))
c2002 + qt(0.975, 8) * sd(residu) * sqrt(1 + 1/10 + (r2002 - mean(revenu))^2/((9)*var(revenu)))
c2003 - qt(0.975, 8) * sd(residu) * sqrt(1 + 1/10 + (r2003 - mean(revenu))^2/((9)*var(revenu)))
c2003 + qt(0.975, 8) * sd(residu) * sqrt(1 + 1/10 + (r2003 - mean(revenu))^2/((9)*var(revenu)))


summary(lm(conso ~ revenu))


# Exercice 4

pere <- c(65,63,67,64,68,62,70,66,68,67,69,71)
fils <- c(68,66,68,65,69,66,68,65,71,67,68,70)
plot(fils ~ pere);abline(0,1)

lm_fils_fct_pere <- lm(fils ~ pere)
summary(lm_fils_fct_pere)

lmèpere_fct_fils <- lm(pere~fils)
summary(lmèpere_fct_fils)
