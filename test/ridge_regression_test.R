library(PeakEngines)

data(longley)
df <- longley
indexes <- which(!names(df) %in% c("Employed"))
df[,indexes] <- scale(df[,indexes])

assert <- function(t) {
  if (!isTRUE(t)) {
    print(t)
    stop("assertion failed")
  }
}

fit <- glmAlo(Employed ~ ., data=df, family=gaussian)
assert(all.equal(fit$lambda, 0.002630110146048188, tolerance=1.0e-3))

expectedCoef <- c(-5.96719296e-03, -1.80931693e+00, -1.64536768e+00, -6.68515327e-01,
                  -8.24883939e-01,  7.41501605e+00)
assert(all.equal(unname(fit$coef[-1]), expectedCoef, tolerance=1.0e-3))

expectedIntercept <- 6.53170000e+01
assert(all.equal(unname(fit$coef[1]), expectedIntercept))

pred1 <- predict(fit, newdata=df[1,], type="response")
expectedPred <- 60.05388412
assert(all.equal(pred1[1], expectedPred))
