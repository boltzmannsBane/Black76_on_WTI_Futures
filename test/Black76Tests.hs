{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import Text.Printf (printf)
import Control.Monad (when)

import Black76

-- ============================
-- Generators
-- ============================

genForward :: Gen Double
genForward = Gen.double (Range.linearFrac 1 200)

genStrike :: Gen Double
genStrike = Gen.double (Range.linearFrac 0.1 200)

genTime :: Gen Double
genTime = Gen.double (Range.linearFrac 0.001 5.0)

genRate :: Gen Double
genRate = Gen.double (Range.linearFrac (-0.05) 0.2)

genVol :: Gen Double
genVol = Gen.double (Range.linearFrac 0.001 2.0)

-- Numeric derivative helpers
centralDiffF :: (Double -> Double) -> Double -> Double -> Double
centralDiffF f x h = (f (x + h) - f (x - h)) / (2 * h)

secondCentralDiffF :: (Double -> Double) -> Double -> Double -> Double
secondCentralDiffF f x h = (f (x + h) - 2 * f x + f (x - h)) / (h * h)

-- Approx equality helper: absolute + relative tolerance
approxEqual :: Double -> Double -> Double -> Bool
approxEqual tol x y =
  let absErr = abs (x - y)
      relErr = if max (abs x) (abs y) <= 1e-12 then 0 else absErr / max (abs x) (abs y)
  in absErr <= tol || relErr <= tol

-- ============================
-- Basic structural properties
-- ============================

prop_call_non_negative :: Property
prop_call_non_negative = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol
  let c = black76Call f k t r sigma
  annotateShow c
  assert (c >= 0)

prop_put_non_negative :: Property
prop_put_non_negative = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol
  let p = black76Put f k t r sigma
  annotateShow p
  assert (p >= 0)

prop_gamma_non_negative :: Property
prop_gamma_non_negative = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol
  let g = black76GammaF f k t r sigma
  annotateShow g
  assert (g >= 0)

prop_vega_non_negative :: Property
prop_vega_non_negative = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol
  let v = black76Vega f k t r sigma
  annotateShow v
  assert (v >= 0)

prop_call_monotone_in_sigma :: Property
prop_call_monotone_in_sigma = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma1 <- forAll genVol
  sigma2 <- forAll genVol

  let sigmaLow  = min sigma1 sigma2
      sigmaHigh = max sigma1 sigma2
      cLow  = black76Call f k t r sigmaLow
      cHigh = black76Call f k t r sigmaHigh

  annotateShow (sigmaLow, sigmaHigh, cLow, cHigh)
  assert (cHigh >= cLow - 1e-12)  -- small tolerance for FP

prop_call_monotone_in_forward :: Property
prop_call_monotone_in_forward = property $ do
  f1 <- forAll genForward
  f2 <- forAll genForward
  k  <- forAll genStrike
  t  <- forAll genTime
  r  <- forAll genRate
  sigma <- forAll genVol

  let fLow  = min f1 f2
      fHigh = max f1 f2
      cLow  = black76Call fLow  k t r sigma
      cHigh = black76Call fHigh k t r sigma

  annotateShow (fLow, fHigh, cLow, cHigh)
  assert (cHigh >= cLow)

-- ============================
-- No-arbitrage & put-call parity
-- ============================

prop_put_call_parity :: Property
prop_put_call_parity = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  let c = black76Call f k t r sigma
      p = black76Put  f k t r sigma
      lhs = c - p
      rhs = exp (-r * t) * (f - k)
      tol = 1e-10
  annotate $ printf "lhs=%.12g rhs=%.12g" lhs rhs
  assert (approxEqual tol lhs rhs)

prop_call_lower_upper_bounds :: Property
prop_call_lower_upper_bounds = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  let c = black76Call f k t r sigma
      lower = exp (-r * t) * max (f - k) 0
      upper = exp (-r * t) * f
      tol = 1e-10
  annotateShow (c, lower, upper)
  assert (c + 1e-12 >= lower - tol)  -- small slack for FP
  assert (c <= upper + tol)

-- ============================
-- Greek consistency (delta/gamma vs finite differences)
-- ============================

prop_delta_vs_numeric :: Property
prop_delta_vs_numeric = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  -- ensure numerically safe
  when (t <= 0 || sigma <= 0) discard

  let callF x = black76Call x k t r sigma
      analyticD = black76DeltaF f k t r sigma
      h = 1e-4 * max 1.0 f
      numericD = centralDiffF callF f h
      tol = 5e-4 + 5e-4 * abs analyticD

  annotate $ printf "analytic=%.12g numeric=%.12g h=%.6g tol=%.6g" analyticD numericD h tol
  assert (abs (analyticD - numericD) <= tol)

prop_gamma_vs_numeric :: Property
prop_gamma_vs_numeric = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  -- Avoid numerically pathological regions
  when (t < 0.05 || sigma < 0.05) discard

  let callF x = black76Call x k t r sigma
      analyticG = black76GammaF f k t r sigma

      -- Step size: scale with f, but not too small
      h = max 1e-4 (1e-3 * f)

      numericG = secondCentralDiffF callF f h

      absTol = 1e-8
      relTol = 1e-2  -- 1%

  annotate $ "analyticG = " <> show analyticG
  annotate $ "numericG  = " <> show numericG
  annotate $ "h         = " <> show h

  -- Near zero: absolute comparison
  if abs analyticG < absTol && abs numericG < absTol
    then success
    else
      let relErr = abs (analyticG - numericG) / max 1e-12 (abs analyticG)
      in assert (relErr <= relTol)

prop_vega_vs_numeric :: Property
prop_vega_vs_numeric = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  -- Avoid numerically pathological regions
  when (t < 0.01 || sigma < 0.01) discard

  let callSigma s = black76Call f k t r s
      analyticV = black76Vega f k t r sigma
      h = 1e-4 * max 0.01 sigma
      numericV = centralDiffF callSigma sigma h
      tol = 5e-4 + 5e-4 * abs analyticV

  annotate $ printf "analyticV=%.12g numericV=%.12g h=%.6g" analyticV numericV h
  assert (abs (analyticV - numericV) <= tol)

-- ============================
-- Limit behavior tests
-- ============================

-- sigma -> 0: price -> discounted intrinsic
prop_sigma_zero_limit :: Property
prop_sigma_zero_limit = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate

  let sigma = 1e-8
      c = black76Call f k t r sigma
      expected = exp (-r * t) * max (f - k) 0
      tol = 1e-8 + 1e-8 * max 1 (abs expected)
  annotate $ printf "c=%.12g expected=%.12g" c expected
  assert (approxEqual tol c expected)

-- sigma -> large: price -> exp(-r t) * f (upper bound)
prop_sigma_large_limit :: Property
prop_sigma_large_limit = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate

  -- choose a large sigma but avoid NaNs/Inf
  let sigma = 1e3  -- very large
      c = black76Call f k t r sigma
      expected = exp (-r * t) * f
      relTol = 1e-6
  annotate $ printf "c=%.12g expected=%.12g" c expected
  -- numeric operations may saturate: allow small absolute or relative slack
  assert (abs (c - expected) <= 1e-6 + relTol * abs expected)

-- t -> 0: price -> discounted intrinsic
prop_time_zero_limit :: Property
prop_time_zero_limit = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  r <- forAll genRate

  let t = 1e-8
      sigma = 0.2  -- arbitrary nonzero vol
      c = black76Call f k t r sigma
      expected = exp (-r * t) * max (f - k) 0
      tol = 1e-8 + 1e-8 * max 1 (abs expected)
  annotate $ printf "t=%.12g c=%.12g expected=%.12g" t c expected
  assert (approxEqual tol c expected)

-- ============================
-- Runner
-- ============================

tests :: IO Bool
tests = checkParallel $ Group "Black-76 Tests"
  [ ("call >= 0", prop_call_non_negative)
  , ("put >= 0", prop_put_non_negative)
  , ("gamma >= 0", prop_gamma_non_negative)
  , ("vega >= 0", prop_vega_non_negative)
  , ("call monotone in F", prop_call_monotone_in_forward)
  , ("call monotone in sigma", prop_call_monotone_in_sigma)
  , ("put-call parity", prop_put_call_parity)
  , ("call bounds (no-arb)", prop_call_lower_upper_bounds)
  , ("delta vs numeric", prop_delta_vs_numeric)
  , ("gamma vs numeric", prop_gamma_vs_numeric)
  , ("vega vs numeric", prop_vega_vs_numeric)
  , ("sigma -> 0 limit", prop_sigma_zero_limit)
  , ("sigma -> large limit", prop_sigma_large_limit)
  , ("t -> 0 limit", prop_time_zero_limit)
  ]

main :: IO ()
main = do
  ok <- tests
  if ok then putStrLn "All tests passed." else fail "Some tests failed."
