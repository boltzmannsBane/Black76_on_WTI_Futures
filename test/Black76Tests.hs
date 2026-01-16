{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import Black76

-- ============================
-- Generators
-- ============================

genForward :: Gen Double
genForward = Gen.double (Range.linearFrac 1 200)

genStrike :: Gen Double
genStrike = Gen.double (Range.linearFrac 1 200)

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

-- ============================
-- Properties
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

-- Optional: delta/gamma finite-difference checks
prop_delta_vs_numeric :: Property
prop_delta_vs_numeric = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol
  -- ensure numerically safe
  if t <= 0 || sigma <= 0 then discard else pure ()

  let callF x = black76Call x k t r sigma
      analyticD = black76DeltaF f k t r sigma
      h = 1e-4 * max 1.0 f
      numericD = centralDiffF callF f h

  annotateShow (analyticD, numericD, h)
  -- tolerance chosen to allow floating error
  assert (abs (analyticD - numericD) <= 5e-4 + 5e-4 * abs analyticD)

prop_gamma_vs_numeric :: Property
prop_gamma_vs_numeric = property $ do
  f <- forAll genForward
  k <- forAll genStrike
  t <- forAll genTime
  r <- forAll genRate
  sigma <- forAll genVol

  -- Avoid singular regimes
  if t < 0.05 || sigma < 0.05 then discard else pure ()

  let callF x = black76Call x k t r sigma
      analyticG = black76GammaF f k t r sigma
      h = max 1e-6 (1e-4 * f)
      numericG = secondCentralDiffF callF f h

  annotateShow (analyticG, numericG, h)

  let relErr = abs (analyticG - numericG) / max 1 (abs analyticG)
  assert (relErr <= 1e-2)


-- ============================
-- Runner
-- ============================

main :: IO ()
main = do
  ok <- checkParallel $ Group "Black-76 Tests"
    [ ("call >= 0", prop_call_non_negative)
    , ("put >= 0", prop_put_non_negative)
    , ("gamma >= 0", prop_gamma_non_negative)
    , ("call monotone in F", prop_call_monotone_in_forward)
    , ("delta vs numeric", prop_delta_vs_numeric)
    , ("gamma vs numeric", prop_gamma_vs_numeric)
    ]
  if ok then putStrLn "All Hedgehog tests passed." else fail "Some Hedgehog tests failed."
