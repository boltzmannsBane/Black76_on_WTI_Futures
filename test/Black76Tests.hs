{-# LANGUAGE RecordWildCards #-}
module Main where

import Test.QuickCheck
import Text.Printf (printf)

import Black76

-- numeric helpers
centralDiffF :: (Double -> Double) -> Double -> Double -> Double
centralDiffF f x h = (f (x + h) - f (x - h)) / (2 * h)

secondCentralDiffF :: (Double -> Double) -> Double -> Double -> Double
secondCentralDiffF f x h = (f (x + h) - 2 * f x + f (x - h)) / (h * h)

-- Generators
genPositiveRange :: Gen Double
genPositiveRange = choose (0.1, 500.0)

genStrike :: Gen Double
genStrike = choose (0.1, 500.0)

genT :: Gen Double
genT = choose (1.0/365.0, 5.0)

genSigma :: Gen Double
genSigma = choose (1e-4, 3.0)

genR :: Gen Double
genR = choose (-0.10, 0.20)

data Params = Params { pF :: Double, pK :: Double, pT :: Double, pR :: Double, pSigma :: Double }
  deriving Show

instance Arbitrary Params where
  arbitrary = do
    f <- genPositiveRange
    k <- genStrike
    t <- genT
    r <- genR
    s <- genSigma
    return $ Params f k t r s

-- Properties (use Black76 functions)
prop_price_nonnegative :: Params -> Bool
prop_price_nonnegative Params{..} =
  black76Call pF pK pT pR pSigma >= -1e-12

prop_monotonic_in_f :: Params -> Positive Double -> Bool
prop_monotonic_in_f Params{..} (Positive delta) =
  let f1 = max 0.1 pF
      f2 = f1 + (abs delta) * 0.5
      c1 = black76Call f1 pK pT pR pSigma
      c2 = black76Call f2 pK pT pR pSigma
  in c2 >= c1 - 1e-10

prop_monotonic_in_sigma :: Params -> Positive Double -> Bool
prop_monotonic_in_sigma Params{..} (Positive bump) =
  let s1 = max 1e-4 pSigma
      s2 = s1 + (abs bump) * 0.01
      c1 = black76Call pF pK pT pR s1
      c2 = black76Call pF pK pT pR s2
  in c2 >= c1 - 1e-10

prop_put_call_parity :: Params -> Bool
prop_put_call_parity Params{..} =
  let c = black76Call pF pK pT pR pSigma
      p = black76Put  pF pK pT pR pSigma
      lhs = c - p
      rhs = exp (-pR * pT) * (pF - pK)
  in abs (lhs - rhs) <= 1e-8 + 1e-8 * abs rhs

prop_delta_bounds :: Params -> Bool
prop_delta_bounds Params{..} =
  let d = black76DeltaF pF pK pT pR pSigma
      ub = exp (-pR * pT)
  in d >= -1e-12 && d <= ub + 1e-12

prop_gamma_nonnegative :: Params -> Bool
prop_gamma_nonnegative Params{..} =
  let g = black76GammaF pF pK pT pR pSigma
  in g >= -1e-14

prop_delta_vs_numeric :: Params -> Property
prop_delta_vs_numeric Params{..} =
  pT > 0 && pSigma > 0 ==>
    let h = 1e-4 * max 1.0 pF
        callF x = black76Call x pK pT pR pSigma
        numericD = centralDiffF callF pF h
        analyticD = black76DeltaF pF pK pT pR pSigma
    in counterexample (printf "analytic=%.8g numeric=%.8g h=%.8g" analyticD numericD h)
         (abs (analyticD - numericD) <= 5e-4 + 5e-4 * abs analyticD)

prop_gamma_vs_numeric :: Params -> Property
prop_gamma_vs_numeric Params{..} =
  pT > 0 && pSigma > 0 ==>
    let h = 1e-3 * max 1.0 pF
        callF x = black76Call x pK pT pR pSigma
        numericGamma = secondCentralDiffF callF pF h
        analyticGamma = black76GammaF pF pK pT pR pSigma
    in counterexample (printf "analytic=%.8g numeric=%.8g h=%.8g" analyticGamma numericGamma h)
         (abs (analyticGamma - numericGamma) <= 5e-3 + 5e-3 * abs analyticGamma)

main :: IO ()
main = do
  putStrLn "Running Black-76 QuickCheck suite..."
  quickCheckWith stdArgs { maxSuccess = 400 } prop_price_nonnegative
  quickCheckWith stdArgs { maxSuccess = 300 } prop_monotonic_in_f
  quickCheckWith stdArgs { maxSuccess = 300 } prop_monotonic_in_sigma
  quickCheckWith stdArgs { maxSuccess = 300 } prop_put_call_parity
  quickCheckWith stdArgs { maxSuccess = 300 } prop_delta_bounds
  quickCheckWith stdArgs { maxSuccess = 300 } prop_gamma_nonnegative
  quickCheckWith stdArgs { maxSuccess = 250 } prop_delta_vs_numeric
  quickCheckWith stdArgs { maxSuccess = 250 } prop_gamma_vs_numeric
  putStrLn "Done."
