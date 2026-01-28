module HedgeSim
  ( -- * Core types
    HedgeState(..)
  , HedgeConfig(..)
  , Price
  , Time
    -- * Original hedging (no transaction costs)
  , stepHedge
  , computePnL
  , payoffCall
    -- * Enhanced hedging with transaction costs
  , stepHedgeWithCosts
  , defaultConfig
    -- * Realized volatility estimation
  , realizedVol
  , rollingRealizedVol
  ) where

import Black76

type Price = Double
type Time  = Double

-- | Hedge simulation state
data HedgeState = HedgeState
  { hsCash      :: Double  -- ^ Cash position (negative = borrowed)
  , hsDelta     :: Double  -- ^ Current hedge ratio
  , hsSpot      :: Double  -- ^ Current underlying price
  , hsTxCosts   :: Double  -- ^ Cumulative transaction costs
  , hsRebalances :: Int    -- ^ Number of rebalances performed
  } deriving Show

-- | Configuration for hedge simulation
data HedgeConfig = HedgeConfig
  { cfgTxCostBps    :: Double  -- ^ Transaction cost in basis points (e.g., 10 = 0.1%)
  , cfgRebalanceThreshold :: Double  -- ^ Only rebalance if |delta change| > threshold
  } deriving Show

-- | Default configuration: 10 bps transaction cost, no threshold
defaultConfig :: HedgeConfig
defaultConfig = HedgeConfig
  { cfgTxCostBps = 10.0
  , cfgRebalanceThreshold = 0.0
  }

-- ============================================================
-- Original stepHedge (backward compatible, no transaction costs)
-- ============================================================

stepHedge
  :: (Double -> Double -> Double -> Double)  -- ^ Delta function: spot -> time -> vol -> delta
  -> Double  -- ^ Strike (unused, for API compatibility)
  -> Double  -- ^ Rate (unused)
  -> Double  -- ^ Volatility
  -> [Price] -- ^ Price series
  -> [Time]  -- ^ Time to expiry series
  -> [HedgeState]
stepHedge _ _ _ _ [] _ = []
stepHedge _ _ _ _ _ [] = []
stepHedge deltaFn _ _ sigma (p0:ps) (t0:ts) =
  scanl go initState (zip ps ts)
  where
    initDelta = deltaFn p0 t0 sigma
    initState = HedgeState
      { hsCash  = -(initDelta * p0)
      , hsDelta = initDelta
      , hsSpot  = p0
      , hsTxCosts = initDelta * p0 * 0  -- No tx costs in original
      , hsRebalances = 0
      }

    go st (s, t) =
      let newDelta = deltaFn s t sigma
          dDelta   = newDelta - hsDelta st
          cash'    = hsCash st - dDelta * s
      in st { hsCash = cash', hsDelta = newDelta, hsSpot = s }

-- ============================================================
-- Enhanced stepHedge with transaction costs and rebalance threshold
-- ============================================================

-- | Step hedge with transaction costs and optional rebalance threshold
--
-- Transaction cost model: proportional cost on notional traded
--   Cost = |ΔDelta| * Spot * (txCostBps / 10000)
--
-- Rebalance threshold: only rebalance if |ΔDelta| > threshold
--   This implements "no-trade bands" to reduce transaction costs
stepHedgeWithCosts
  :: HedgeConfig
  -> (Double -> Double -> Double -> Double)  -- ^ Delta function
  -> Double  -- ^ Volatility (or use realized vol)
  -> [Price] -- ^ Price series
  -> [Time]  -- ^ Time to expiry series
  -> [HedgeState]
stepHedgeWithCosts _ _ _ [] _ = []
stepHedgeWithCosts _ _ _ _ [] = []
stepHedgeWithCosts cfg deltaFn sigma (p0:ps) (t0:ts) =
  scanl go initState (zip ps ts)
  where
    txCostRate = cfgTxCostBps cfg / 10000.0
    threshold  = cfgRebalanceThreshold cfg

    initDelta = deltaFn p0 t0 sigma
    initTxCost = abs initDelta * p0 * txCostRate  -- Cost of initial hedge

    initState = HedgeState
      { hsCash  = -(initDelta * p0) - initTxCost
      , hsDelta = initDelta
      , hsSpot  = p0
      , hsTxCosts = initTxCost
      , hsRebalances = 0
      }

    go st (s, t) =
      let targetDelta = deltaFn s t sigma
          dDelta = targetDelta - hsDelta st
      in if abs dDelta <= threshold
         -- Don't rebalance - just update spot
         then st { hsSpot = s }
         -- Rebalance with transaction costs
         else let txCost = abs dDelta * s * txCostRate
                  cash'  = hsCash st - dDelta * s - txCost
              in HedgeState
                   { hsCash = cash'
                   , hsDelta = targetDelta
                   , hsSpot = s
                   , hsTxCosts = hsTxCosts st + txCost
                   , hsRebalances = hsRebalances st + 1
                   }

-- ============================================================
-- P&L calculation
-- ============================================================

computePnL :: HedgeState -> Double -> Double
computePnL st payoff =
  hsCash st + hsDelta st * hsSpot st - payoff

payoffCall :: Double -> Double -> Double
payoffCall k s = max 0 (s - k)

-- ============================================================
-- Realized Volatility Estimation
-- ============================================================

-- | Calculate annualized realized volatility from log returns
--
-- Formula: σ_realized = sqrt(252) * std(log returns)
--
-- This uses the standard "close-to-close" estimator.
-- More sophisticated estimators (Parkinson, Garman-Klass) would use OHLC data.
realizedVol :: [Price] -> Double
realizedVol prices
  | length prices < 2 = 0
  | otherwise =
      let logReturns = zipWith (\p1 p2 -> log (p2 / p1)) prices (tail prices)
          n = length logReturns
          mean_r = sum logReturns / fromIntegral n
          variance = sum [(r - mean_r)^(2::Int) | r <- logReturns] / fromIntegral (n - 1)
          dailyVol = sqrt variance
      in dailyVol * sqrt 252  -- Annualize assuming 252 trading days

-- | Rolling realized volatility with specified window
--
-- Returns a list of (time_index, realized_vol) pairs.
-- The first (window-1) points have no vol estimate.
rollingRealizedVol :: Int -> [Price] -> [(Int, Double)]
rollingRealizedVol window prices
  | window < 2 = []
  | length prices < window = []
  | otherwise =
      [ (i, realizedVol (take window $ drop (i - window + 1) prices))
      | i <- [window - 1 .. length prices - 1]
      ]
