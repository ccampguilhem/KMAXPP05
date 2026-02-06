library(tidyverse)

# Exercice 1)
x<-(-50:50)/10
sigma_fct <- function(t){exp(t)/(1 + exp(t))}
y <- sigma_fct(x)
plot(y~x)

sigma_fct(c(-5,-1,-2,8))
inv_sigma <- function(lambda){log(lambda/(1-lambda))}
inv_sigma(sigma_fct(c(-2,-1)))


#Exercice 6)
ytest <- c(0,0,1,0,1)
p_xtest <- c(0.8,0.1,0.5,0.4,0.9)

clf <- function(p, lambda){
  (p >= lambda)*1
}
hat_y_0.5 = clf(p = p_xtest, lambda = 0.5)
table(ytest, hat_y_0.5)

TDP <- function(hat_y, y){
  sum((hat_y == 1) * (y == 1))/sum(y == 1)
}
TDP(hat_y = hat_y_0.5, y = ytest)
FDP <- function(hat_y, y){
  sum((hat_y == 1) * (y == 0))/sum(y == 0)
}
FDP(hat_y = hat_y_0.5, y = ytest)

lambdas <- (0:1000)/1000
df <- data.frame()
for(lambda in lambdas){
  hat_y <- clf(p = p_xtest, lambda = lambda)
  df<- rbind(df, data.frame(lambda = lambda,
                  FDP = FDP(hat_y, ytest),
                  TDP = TDP(hat_y, ytest) ) )
}

ggplot(df, aes(x = lambda, y = TDP)) + geom_line() + geom_point()
ggplot(df, aes(x = lambda, y = FDP)) + geom_line() + geom_point()
ggplot(df, aes(x = FDP, y = TDP)) + geom_point()

#exercice 7)
#Exercice 6)
ytest <- c(0,0,1,0,1,1,1)
p_xtest <- c(0.8,0.1,0.5,0.4,0.9, 0.3,0.7)

clf <- function(p, lambda){
  (p >= lambda)*1
}
hat_y_0.5 = clf(p = p_xtest, lambda = 0.5)
table(ytest, hat_y_0.5)

TDP <- function(hat_y, y){
  sum((hat_y == 1) * (y == 1))/sum(y == 1)
}
TDP(hat_y = hat_y_0.5, y = ytest)
FDP <- function(hat_y, y){
  sum((hat_y == 1) * (y == 0))/sum(y == 0)
}
FDP(hat_y = hat_y_0.5, y = ytest)

lambdas <- (0:10)/10
df <- data.frame()
for(lambda in lambdas){
  hat_y <- clf(p = p_xtest, lambda = lambda)
  df<- rbind(df, data.frame(lambda = lambda,
                            FDP = FDP(hat_y, ytest),
                            TDP = TDP(hat_y, ytest) ) )
}

ggplot(df, aes(x = lambda, y = TDP)) + geom_line() + geom_point()
ggplot(df, aes(x = lambda, y = FDP)) + geom_line() + geom_point()
ggplot(df, aes(x = FDP, y = TDP)) + geom_point() + geom_line()


# Installer si nécessaire
install.packages("pROC")

# Charger le package
library(pROC)

# Génération de données simulées
set.seed(123)
y_true <- ytest  # Classes (0 = négatif, 1 = positif)
y_scores <- p_xtest  # Scores prédits (probabilités)

# Calcul de la courbe ROC
roc_obj <- roc(y_true, y_scores)

# Affichage de la courbe ROC
plot(roc_obj, col = "blue", lwd = 2, main = "Courbe ROC")
auc(roc_obj)  # Calcul de l'AUC



#exercice 7
xtrain <- matrix(c(0,0,0,2,2,2,1,0,-1,-2,1,-2), ncol = 2, byrow = TRUE)
ytrain <- c(1,1,1,0,0,0)

xtest <- matrix(c(-1,-1,1,1,1,2,2,-1,1,-1,1,2,-1,1), ncol = 2)
ytest <- c(1,1,0,1,0,0,0)

t = 0.5 + xtest %*% c(-1,1)
proba_t_lr <- (exp(t))/(1+exp(t))



library(proxy)

# Matrice des distances euclidiennes entre toutes les lignes de A et B
dist_matrix <- proxy::dist(xtrain, xtest, method = "Euclidean")
print(as.matrix(dist_matrix))  # Convertir en matrice pour l'affichage
third_min <- apply(t(dist_matrix), 1, function(col) sort(col, partial = 3)[3])
print(dist_matrix)
print(third_min)

indiv_min <- (t(dist_matrix) <= third_min)

proba_knn <- indiv_min %*% ytrain / rowSums(indiv_min*1)

roc_curve <- function(ypred, ytest, nb_lambdas = 1000, name_method){
  lambdas <- seq(from = 0, to = 1,length.out = nb_lambdas)
  df <- data.frame()
  for(lambda in lambdas){
    hat_y <- clf(p = ypred, lambda = lambda)
    df<- rbind(df, data.frame(lambda = lambda,
                              FDP = FDP(hat_y, ytest),
                              TDP = TDP(hat_y, ytest),
                              method = name_method) )
  }
  return(df)
}
df_lr <- roc_curve(proba_t_lr, ytest, nb_lambdas = 11, name_method = "lr")
df_knn <- roc_curve(proba_knn, ytest, name_method = "knn")
df <- rbind(df_lr, df_knn)
ggplot(df, aes(x = FDP, y = TDP, color = method)) +
  geom_point() #+ geom_line()

# Calcul de la courbe ROC
roc_obj <- roc(ytest, proba_t_lr)

# Affichage de la courbe ROC
plot(roc_obj, col = "blue", lwd = 2, main = "Courbe ROC")
auc(roc_obj)  # Calcul de l'AUC


# Calcul de la courbe ROC
roc_obj_knn <- roc(ytest, proba_knn)

# Affichage de la courbe ROC
plot(roc_obj_knn, col = "red", lwd = 2, main = "Courbe ROC", add= TRUE)
auc(roc_obj_knn)  # Calcul de l'AUC
