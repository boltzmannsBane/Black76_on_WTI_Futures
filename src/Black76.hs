module Black76
  ( black76Call
  , black76Put
  , black76DeltaF
  , black76GammaF
  ) where

import Statistics.Distribution (cumulative)
import Statistics.Distribution.Normal (normalDistr)

black76Call :: Double -> Double -> Double -> Double -> Double -> Double
black76Call f k t r sigma
    | t <= 0    = max 0 (f - k)
    | otherwise = exp (-r * t) * (f * normCdf d1 - k * normCdf d2)
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom
    d2 = d1 - denom
    normCdf x = cumulative (normalDistr 0 1) x

black76Put :: Double -> Double -> Double -> Double -> Double -> Double
black76Put f k t r sigma
    | t <= 0    = max 0 (k - f)
    | otherwise = exp (-r * t) * (k * normCdf (-d2) - f * normCdf (-d1))
  where
    denom = sigma * sqrt t
    d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom
    d2 = d1 - denom
    normCdf x = cumulative (normalDistr 0 1) x

black76DeltaF :: Double -> Double -> Double -> Double -> Double -> Double
black76DeltaF f k t r sigma
  | t <= 0    = exp (-r * t) * (if f > k then 1 else 0)
  | otherwise = exp (-r * t) * normCdf d1
  where
    (d1, _) = d1d2 f k t sigma
    normCdf x = cumulative (normalDistr 0 1) x

black76GammaF :: Double -> Double -> Double -> Double -> Double -> Double
black76GammaF f k t r sigma
  | t <= 0    = 0
  | otherwise = exp (-r * t) * normPdf d1 / (f * sigma * sqrt t)
  where
    (d1, _) = d1d2 f k t sigma

-- Helpers

d1d2 :: Double -> Double -> Double -> Double -> (Double, Double)
d1d2 f k t sigma =
  let denom = sigma * sqrt t
      d1 = (log (f / k) + 0.5 * sigma * sigma * t) / denom
      d2 = d1 - denom
  in (d1, d2)

normPdf :: Double -> Double
normPdf x = exp (-0.5 * x * x) / sqrt (2 * pi)
