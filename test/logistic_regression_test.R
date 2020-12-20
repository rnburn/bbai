library(PeakEngines)

data("PimaIndiansDiabetes2", package = "mlbench")

df <- na.omit(PimaIndiansDiabetes2)
indexes <- which(!names(df) %in% c("diabetes"))
df[,indexes] <- scale(df[,indexes])

assert <- function(t) {
  if (!isTRUE(t)) {
    print(t)
    stop("assertion failed")
  }
}

fit <- glmAlo(diabetes ~ ., data=df, family=binomial)
assert(all.equal(fit$lambda, 4.196021915391197, tolerance=1.0e-3))

expectedCoef <- c(0.23222633, 0.94366593, 0.02972366, 0.15297691, 0.01861822,
        0.37318937, 0.32387612, 0.31800703)
assert(all.equal(unname(fit$coef[-1]), expectedCoef, tolerance=1.0e-3))

expectedIntercept <- -0.94747798
assert(all.equal(unname(fit$coef[1]), expectedIntercept, tolerance=1.0e-3))

pred1 <- predict(fit, newdata=df[1,], type="response")
expectedPred <- 0.04076011
assert(all.equal(pred1[1], expectedPred, tolerance=1.0e-3))

pred1 <- predict(fit, newdata=df[1,], type="link")
expectedPred <- -3.158437282512427
assert(all.equal(pred1[1], expectedPred, tolerance=1.0e-3))
