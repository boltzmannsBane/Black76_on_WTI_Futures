module Main where

import HedgeSim
import Black76
import DataLoader
import MojoBridge
import System.IO

main :: IO ()
main = do
  -- Load real WTI futures prices
  pricePoints <- loadWTIPrices "wti_futures.csv"
  let prices = pricesToList pricePoints
      n = length prices

      -- Option parameters
      k = 60.0          -- Strike price (near current WTI levels)
      r = 0.05          -- Risk-free rate
      sigma = 0.25      -- Implied volatility (~25% for WTI)

      -- Generate time-to-expiry series (daily, 252 trading days/year)
      -- Assume 1 year expiry from start, decreasing daily
      dt = 1.0 / 252.0
      times = [1.0 - dt * fromIntegral i | i <- [0..n-1]]

      -- Delta function for hedging
      deltaFn s t sig = black76DeltaF s k t r sig

      -- Run hedge simulation
      states = stepHedge deltaFn k r sigma prices times

      finalSpot = last prices
      payoff = payoffCall k finalSpot

      pnls = map (\st -> computePnL st payoff) states

  putStrLn $ "Loaded " ++ show n ++ " price points"
  putStrLn $ "Date range: " ++ ppDate (head pricePoints) ++ " to " ++ ppDate (last pricePoints)
  putStrLn $ "Price range: $" ++ show (minimum prices) ++ " - $" ++ show (maximum prices)
  putStrLn $ "Strike: $" ++ show k
  putStrLn $ "Final spot: $" ++ show finalSpot
  putStrLn $ "Option payoff: $" ++ show payoff
  putStrLn $ "Final hedge P&L: $" ++ show (last pnls)

  -- Write P&L timeseries to CSV
  writeFile "pnl.csv" $
    unlines $ "step,pnl" : zipWith (\i p -> show i ++ "," ++ show p) [0..] pnls

  putStrLn "Wrote pnl.csv"

-- Run Mojo Monte Carlo for tail risk analysis
runMojo :: IO ()
runMojo = do
  let input = MojoInput
        { s0 = 60        -- Current WTI price level
        , mu = 0
        , sigma = 0.25   -- 25% annualized vol
        , r = 0.05
        , k = 60
        , steps = 252    -- Daily steps for 1 year
        , paths = 100000
        }

  result <- runMojoMC input
  print result
