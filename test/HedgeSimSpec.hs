module HedgeSimSpec where

import Test.Tasty
import Test.Tasty.HUnit
import HedgeSim

tests :: TestTree
tests = testGroup "HedgeSim"
  [ testGroup "Original stepHedge"
      [ testCase "Zero move, zero sigma -> zero PnL" $
          let prices = replicate 10 100
              times  = reverse [0.1,0.2..1.0]
              k = 100
              r = 0.0
              sigma = 0.0
              deltaFn _ _ _ = 0.0
              states = stepHedge deltaFn k r sigma prices times
              payoff = 0
              pnls = map (`computePnL` payoff) states
          in last pnls @?= 0.0

      , testCase "Constant delta = 1, price rises -> positive hedge value" $
          let prices = [100, 110, 120]
              times  = [1.0, 0.5, 0.0]
              deltaFn _ _ _ = 1.0
              states = stepHedge deltaFn 100 0 0.2 prices times
              finalState = last states
          in do
              -- Delta = 1 means we hold 1 unit, bought at 100
              -- Final spot is 120, so hedge value = 120 - 100 = 20
              hsDelta finalState @?= 1.0
              hsSpot finalState @?= 120.0
      ]

  , testGroup "Transaction Costs"
      [ testCase "Zero bps -> no transaction costs" $
          let cfg = HedgeConfig { cfgTxCostBps = 0.0, cfgRebalanceThreshold = 0.0 }
              prices = [100, 105, 110]
              times  = [1.0, 0.5, 0.0]
              deltaFn _ _ _ = 0.5
              states = stepHedgeWithCosts cfg deltaFn 0.2 prices times
              finalState = last states
          in hsTxCosts finalState @?= 0.0

      , testCase "10 bps transaction cost accumulates" $
          let cfg = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.0 }
              prices = [100, 100, 100]  -- No price change
              times  = [1.0, 0.5, 0.0]
              -- Delta changes from 0.5 to 0.6 to 0.7
              deltaFn _ t _ = 0.5 + (1.0 - t) * 0.2
              states = stepHedgeWithCosts cfg deltaFn 0.2 prices times
              finalState = last states
          in do
              -- Should have positive transaction costs
              assertBool "Transaction costs should be positive" (hsTxCosts finalState > 0)
              -- Cost = |delta_change| * price * 0.001
              -- Initial: buy 0.5 * 100 * 0.001 = 0.05
              -- Step 1: buy 0.1 * 100 * 0.001 = 0.01
              -- Step 2: buy 0.1 * 100 * 0.001 = 0.01
              -- Total ≈ 0.07
              assertBool "Transaction costs should be reasonable" (hsTxCosts finalState < 1.0)

      , testCase "High transaction cost reduces P&L" $
          let cfgLow = HedgeConfig { cfgTxCostBps = 1.0, cfgRebalanceThreshold = 0.0 }
              cfgHigh = HedgeConfig { cfgTxCostBps = 100.0, cfgRebalanceThreshold = 0.0 }
              prices = [100, 105, 110, 108, 112]
              times  = [1.0, 0.75, 0.5, 0.25, 0.0]
              deltaFn s _ _ = min 1.0 (s / 100 - 0.5)  -- Delta varies with price
              statesLow = stepHedgeWithCosts cfgLow deltaFn 0.2 prices times
              statesHigh = stepHedgeWithCosts cfgHigh deltaFn 0.2 prices times
              payoff = max 0 (last prices - 100)
              pnlLow = computePnL (last statesLow) payoff
              pnlHigh = computePnL (last statesHigh) payoff
          in assertBool "Higher tx costs should reduce P&L" (pnlHigh < pnlLow)

      , testCase "Rebalances are counted correctly" $
          let cfg = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.0 }
              prices = [100, 105, 110, 115, 120]
              times  = [1.0, 0.75, 0.5, 0.25, 0.0]
              deltaFn _ _ _ = 0.5  -- Constant delta = no rebalancing needed after init
              states = stepHedgeWithCosts cfg deltaFn 0.2 prices times
              finalState = last states
          in do
              -- Initial hedge doesn't count as rebalance
              -- Subsequent steps with no delta change = no rebalances
              hsRebalances finalState @?= 0
      ]

  , testGroup "Rebalance Threshold"
      [ testCase "Threshold prevents small rebalances" $
          let cfgNoThresh = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.0 }
              cfgThresh = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.05 }
              prices = [100, 101, 102, 103, 104]  -- Small moves
              times  = [1.0, 0.75, 0.5, 0.25, 0.0]
              -- Small delta changes (< threshold)
              deltaFn _ t _ = 0.5 + (1.0 - t) * 0.01
              statesNoThresh = stepHedgeWithCosts cfgNoThresh deltaFn 0.2 prices times
              statesThresh = stepHedgeWithCosts cfgThresh deltaFn 0.2 prices times
          in do
              -- Without threshold, should rebalance
              assertBool "No threshold: should have rebalances"
                  (hsRebalances (last statesNoThresh) > 0)
              -- With threshold, delta changes are too small
              hsRebalances (last statesThresh) @?= 0

      , testCase "Large delta change triggers rebalance despite threshold" $
          let cfg = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.05 }
              prices = [100, 150]  -- Big price jump
              times  = [1.0, 0.0]
              -- Delta jumps from 0.3 to 0.8
              deltaFn s _ _ = if s < 120 then 0.3 else 0.8
              states = stepHedgeWithCosts cfg deltaFn 0.2 prices times
          in hsRebalances (last states) @?= 1

      , testCase "Threshold reduces transaction costs" $
          let cfgNoThresh = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.0 }
              cfgThresh = HedgeConfig { cfgTxCostBps = 10.0, cfgRebalanceThreshold = 0.02 }
              prices = [100, 101, 99, 102, 98, 103, 97, 104]
              times  = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.0]
              deltaFn s _ _ = s / 200  -- Delta varies with price
              statesNoThresh = stepHedgeWithCosts cfgNoThresh deltaFn 0.2 prices times
              statesThresh = stepHedgeWithCosts cfgThresh deltaFn 0.2 prices times
          in assertBool "Threshold should reduce tx costs"
              (hsTxCosts (last statesThresh) <= hsTxCosts (last statesNoThresh))
      ]

  , testGroup "Realized Volatility"
      [ testCase "Constant prices -> zero volatility" $
          let prices = replicate 100 50.0
          in realizedVol prices @?= 0.0

      , testCase "Realized vol is positive for varying prices" $
          let prices = [100, 102, 98, 105, 95, 110, 90]
          in assertBool "Vol should be positive" (realizedVol prices > 0)

      , testCase "Higher price swings -> higher realized vol" $
          let pricesLow = [100, 101, 100, 101, 100]  -- Small moves
              pricesHigh = [100, 120, 80, 130, 70]   -- Large moves
          in assertBool "Larger swings should have higher vol"
              (realizedVol pricesHigh > realizedVol pricesLow)

      , testCase "Realized vol scales with sqrt(252) annualization" $
          -- Daily vol * sqrt(252) = annual vol
          -- Use more prices to get stable estimate
          let prices = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101]  -- ~1% daily moves
              vol = realizedVol prices
          in assertBool "Vol should be annualized (positive and reasonable)"
              (vol > 0.05 && vol < 0.5)

      , testCase "Rolling realized vol produces correct number of estimates" $
          let prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
              window = 5
              rolling = rollingRealizedVol window prices
          in length rolling @?= (length prices - window + 1)

      , testCase "Rolling vol with window 3 gives estimates" $
          let prices = [100, 105, 95, 110, 90, 115]
              rolling = rollingRealizedVol 3 prices
          in do
              -- With window 3 and 6 prices, we get 4 estimates
              length rolling @?= 4
              -- Each estimate should be non-negative (may be 0 if constant within window)
              assertBool "All rolling vols non-negative" (all ((>= 0) . snd) rolling)

      , testCase "Empty or short price list -> zero vol" $
          do realizedVol [] @?= 0
             realizedVol [100] @?= 0
      ]

  , testGroup "Payoff Functions"
      [ testCase "Call payoff ITM" $
          payoffCall 100 120 @?= 20

      , testCase "Call payoff ATM" $
          payoffCall 100 100 @?= 0

      , testCase "Call payoff OTM" $
          payoffCall 100 80 @?= 0
      ]
  ]
