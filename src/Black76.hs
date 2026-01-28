module Black76
  ( black76Call
  , black76Put
  , black76DeltaF
  , black76GammaF
  , black76Vega
  ) where

black76Call :: Double -> Double -> Double -> Double -> Double -> Double
black76Call f k t r sigma
    | t <= 0    = max 0 (f - k)
    | otherwise = max 0 $
        exp (-r * t) * (f * normCDF d1 - k * normCDF d2)
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom
    d2 = d1 - denom

black76Put :: Double -> Double -> Double -> Double -> Double -> Double
black76Put f k t r sigma
    | t <= 0    = max 0 (k - f)
    | otherwise = max 0 $
        exp (-r * t) * (k * normCDF (-d2) - f * normCDF (-d1))
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom
    d2 = d1 - denom

black76DeltaF :: Double -> Double -> Double -> Double -> Double -> Double
black76DeltaF f k t r sigma
  | t <= 0    = exp (-r * t) * if f > k then 1 else 0
  | otherwise = exp (-r * t) * normCDF d1
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom

black76GammaF :: Double -> Double -> Double -> Double -> Double -> Double
black76GammaF f k t r sigma
  | t <= 0    = 0
  | otherwise =
      exp (-r * t) * normPdf d1 / (f * sigma * sqrt t)
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom

-- | Vega: sensitivity of option price to volatility
-- Vega = e^(-rT) * F * sqrt(T) * N'(d1)
-- Same for calls and puts
black76Vega :: Double -> Double -> Double -> Double -> Double -> Double
black76Vega f k t r sigma
  | t <= 0    = 0
  | otherwise =
      exp (-r * t) * f * sqrt t * normPdf d1
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom

-- ======================================================
-- Normal distribution (no external dependencies)
-- ======================================================

normPdf :: Double -> Double
normPdf x = exp (-0.5 * x * x) / sqrt (2 * pi)

normCDF :: Double -> Double
normCDF x = 0.5 * (1.0 + erf (x / sqrt 2))

erf :: Double -> Double
erf z =
  let t = 1.0 / (1.0 + 0.5 * abs z)
      tau =
        t * exp
          ( -z*z - 1.26551223
          + t * ( 1.00002368
          + t * ( 0.37409196
          + t * ( 0.09678418
          + t * (-0.18628806
          + t * ( 0.27886807
          + t * (-1.13520398
          + t * ( 1.48851587
          + t * (-0.82215223
          + t *   0.17087277)))))))))
  in if z >= 0 then 1 - tau else tau - 1
