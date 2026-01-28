module Main where

import HedgeSim
import Black76
import DataLoader
import MojoBridge
import Text.Printf (printf)

main :: IO ()
main = do
  -- Load real WTI futures prices
  pricePoints <- loadWTIPrices "wti_futures.csv"
  let prices = pricesToList pricePoints
      n = length prices

      -- Option parameters
      k = 60.0          -- Strike price (near current WTI levels)
      r = 0.05          -- Risk-free rate

      -- Calculate realized volatility from historical data
      realizedSigma = realizedVol prices

      -- Also use assumed vol for comparison
      assumedSigma = 0.25

      -- Generate time-to-expiry series (daily, 252 trading days/year)
      dt = 1.0 / 252.0
      times = [1.0 - dt * fromIntegral i | i <- [0..n-1]]

      -- Delta function for hedging
      deltaFn s t sig = black76DeltaF s k t r sig

      finalSpot = last prices
      payoff = payoffCall k finalSpot

  putStrLn "=============================================="
  putStrLn "  WTI Futures Delta Hedge Simulation"
  putStrLn "=============================================="
  putStrLn ""
  putStrLn $ "Loaded " ++ show n ++ " price points"
  putStrLn $ "Date range: " ++ ppDate (head pricePoints) ++ " to " ++ ppDate (last pricePoints)
  printf "Price range: $%.2f - $%.2f\n" (minimum prices) (maximum prices)
  printf "Strike: $%.2f\n" k
  printf "Final spot: $%.2f\n" finalSpot
  printf "Option payoff: $%.2f\n" payoff
  putStrLn ""

  -- Realized vs assumed volatility
  putStrLn "--- Volatility Analysis ---"
  printf "Realized volatility (historical): %.2f%%\n" (realizedSigma * 100)
  printf "Assumed volatility: %.2f%%\n" (assumedSigma * 100)
  printf "Vol misspecification: %.2f%%\n" ((realizedSigma - assumedSigma) * 100)
  putStrLn ""

  -- ============================================================
  -- Hedge Simulation 1: No transaction costs (original)
  -- ============================================================
  putStrLn "--- Hedge Simulation: No Transaction Costs ---"
  let statesNoTx = stepHedge deltaFn k r assumedSigma prices times
      pnlsNoTx = map (`computePnL` payoff) statesNoTx
      finalStateNoTx = last statesNoTx
  printf "Final P&L: $%.2f\n" (last pnlsNoTx)
  printf "Rebalances: %d (daily)\n" (n - 1)
  putStrLn ""

  -- ============================================================
  -- Hedge Simulation 2: With transaction costs (10 bps)
  -- ============================================================
  putStrLn "--- Hedge Simulation: With Transaction Costs (10 bps) ---"
  let cfg10bps = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.0 }
      statesTx10 = stepHedgeWithCosts cfg10bps deltaFn assumedSigma prices times
      pnlsTx10 = map (`computePnL` payoff) statesTx10
      finalStateTx10 = last statesTx10
  printf "Final P&L: $%.2f\n" (last pnlsTx10)
  printf "Total transaction costs: $%.2f\n" (hsTxCosts finalStateTx10)
  printf "Rebalances: %d\n" (hsRebalances finalStateTx10)
  putStrLn ""

  -- ============================================================
  -- Hedge Simulation 3: With transaction costs + threshold (reduce trading)
  -- ============================================================
  putStrLn "--- Hedge Simulation: With Costs + Rebalance Threshold (delta > 0.01) ---"
  let cfgThreshold = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.01 }
      statesThresh = stepHedgeWithCosts cfgThreshold deltaFn assumedSigma prices times
      pnlsThresh = map (`computePnL` payoff) statesThresh
      finalStateThresh = last statesThresh
  printf "Final P&L: $%.2f\n" (last pnlsThresh)
  printf "Total transaction costs: $%.2f\n" (hsTxCosts finalStateThresh)
  printf "Rebalances: %d (reduced from %d)\n" (hsRebalances finalStateThresh) (n - 1)
  putStrLn ""

  -- ============================================================
  -- Hedge Simulation 4: Using realized volatility
  -- ============================================================
  putStrLn "--- Hedge Simulation: Using Realized Volatility ---"
  let statesRealVol = stepHedge deltaFn k r realizedSigma prices times
      pnlsRealVol = map (`computePnL` payoff) statesRealVol
  printf "Final P&L: $%.2f\n" (last pnlsRealVol)
  printf "Difference from assumed vol: $%.2f\n" (last pnlsRealVol - last pnlsNoTx)
  putStrLn ""

  -- Write P&L timeseries to CSV (with all variants)
  writeFile "pnl.csv" $
    unlines $ "step,pnl_no_tx,pnl_tx_10bps,pnl_threshold,pnl_real_vol" :
      [ show i ++ "," ++ show p1 ++ "," ++ show p2 ++ "," ++ show p3 ++ "," ++ show p4
      | (i, (p1, p2, p3, p4)) <- zip [0::Int ..] (zip4 pnlsNoTx pnlsTx10 pnlsThresh pnlsRealVol)
      ]
  putStrLn "Wrote pnl.csv (with all hedge variants)"
  putStrLn ""

  -- ============================================================
  -- Monte Carlo: Compare GBM, Jump-Diffusion, Stochastic Vol
  -- ============================================================
  putStrLn "=============================================="
  putStrLn "  Monte Carlo Tail Risk Analysis"
  putStrLn "=============================================="
  putStrLn ""

  -- GBM (baseline)
  putStrLn "--- Model: GBM (baseline) ---"
  runMojoGBM

  -- Jump-Diffusion
  putStrLn "--- Model: Jump-Diffusion (Merton) ---"
  runMojoJump

  -- Stochastic Volatility
  putStrLn "--- Model: Stochastic Volatility (Heston-like) ---"
  runMojoStochVol

-- Helper to zip 4 lists
zip4 :: [a] -> [b] -> [c] -> [d] -> [(a, b, c, d)]
zip4 (a:as) (b:bs) (c:cs) (d:ds) = (a, b, c, d) : zip4 as bs cs ds
zip4 _ _ _ _ = []

-- ============================================================
-- Monte Carlo runners for different models
-- ============================================================

runMojoGBM :: IO ()
runMojoGBM = do
  let input = MojoInput
        { s0 = 60
        , mu = 0
        , sigma = 0.25
        , r = 0.05
        , k = 60
        , steps = 252
        , paths = 100000
        }
  result <- runMojoMC input
  printMCResult result

runMojoJump :: IO ()
runMojoJump = do
  let input = MojoInputExt
        { extS0 = 60
        , extMu = 0
        , extSigma = 0.25
        , extR = 0.05
        , extK = 60
        , extSteps = 252
        , extPaths = 100000
        , extModel = JumpDiffusion
        , extJump = Just JumpParams
            { jpIntensity = 2.0   -- 2 jumps per year
            , jpMean = -0.05      -- 5% down-jump on average
            , jpVol = 0.15        -- 15% jump vol
            }
        , extStochVol = Nothing
        }
  result <- runMojoMCExt input
  printMCResult result

runMojoStochVol :: IO ()
runMojoStochVol = do
  let input = MojoInputExt
        { extS0 = 60
        , extMu = 0
        , extSigma = 0.25
        , extR = 0.05
        , extK = 60
        , extSteps = 252
        , extPaths = 100000
        , extModel = StochasticVol
        , extJump = Nothing
        , extStochVol = Just StochVolParams
            { svMeanRev = 2.0       -- Mean reversion speed
            , svLongTermVol = 0.25  -- Long-term 25% vol
            , svVolOfVol = 0.4      -- 40% vol-of-vol
            , svCorrelation = -0.7  -- Negative correlation (leverage effect)
            }
        }
  result <- runMojoMCExt input
  printMCResult result

printMCResult :: MojoOutput -> IO ()
printMCResult result = do
  printf "  Mean payoff: $%.2f\n" (mean result)
  printf "  Std dev:     $%.2f\n" (std result)
  printf "  VaR (99%%):   $%.2f\n" (var_99 result)
  printf "  CVaR (99%%):  $%.2f\n" (cvar_99 result)
  putStrLn ""
