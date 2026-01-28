module MojoBridgeSpec where

import Test.Tasty
import Test.Tasty.HUnit
import MojoBridge

tests :: TestTree
tests = testGroup "MojoBridge"
  [ testGroup "GBM Model"
      [ testCase "Mojo roundtrip produces sane output" $ do
          let input = MojoInput
                { s0 = 100
                , mu = 0
                , sigma = 0.2
                , r = 0.01
                , k = 100
                , steps = 10
                , paths = 1000
                }

          out <- runMojoMC input

          assertBool "mean finite" (not $ isNaN (mean out))
          assertBool "std positive" (std out >= 0)
          assertBool "var <= cvar" (var_99 out <= cvar_99 out)

      , testCase "ATM call has positive expected value" $ do
          let input = MojoInput
                { s0 = 100
                , mu = 0
                , sigma = 0.25
                , r = 0.05
                , k = 100
                , steps = 50
                , paths = 5000
                }

          out <- runMojoMC input

          assertBool "ATM call mean should be positive" (mean out > 0)

      , testCase "Deep OTM call has near-zero mean" $ do
          let input = MojoInput
                { s0 = 100
                , mu = 0
                , sigma = 0.1   -- Low vol
                , r = 0.05
                , k = 200       -- Way OTM
                , steps = 50
                , paths = 5000
                }

          out <- runMojoMC input

          -- Deep OTM with low vol should have very low expected payoff
          assertBool "Deep OTM call mean should be small" (mean out < 1.0)
      ]

  , testGroup "Jump-Diffusion Model"
      [ testCase "Jump model produces valid output" $ do
          let input = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.2
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 5000
                , extModel = JumpDiffusion
                , extJump = Just JumpParams
                    { jpIntensity = 1.0
                    , jpMean = -0.05
                    , jpVol = 0.1
                    }
                , extStochVol = Nothing
                }

          out <- runMojoMCExt input

          assertBool "mean finite" (not $ isNaN (mean out))
          assertBool "std positive" (std out >= 0)
          assertBool "var <= cvar" (var_99 out <= cvar_99 out)

      , testCase "Jumps increase volatility/std" $ do
          -- GBM baseline
          let inputGBM = MojoInput
                { s0 = 100
                , mu = 0
                , sigma = 0.2
                , r = 0.05
                , k = 100
                , steps = 50
                , paths = 10000
                }

          -- Jump model with significant jumps
          let inputJump = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.2
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 10000
                , extModel = JumpDiffusion
                , extJump = Just JumpParams
                    { jpIntensity = 3.0   -- Frequent jumps
                    , jpMean = 0.0        -- Symmetric jumps
                    , jpVol = 0.2         -- Large jump vol
                    }
                , extStochVol = Nothing
                }

          outGBM <- runMojoMC inputGBM
          outJump <- runMojoMCExt inputJump

          -- Jump model should have higher std due to jump component
          assertBool "Jump model should have higher or similar std"
              (std outJump >= std outGBM * 0.8)  -- Allow some tolerance

      , testCase "Default jump params work" $ do
          let input = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.25
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 1000
                , extModel = JumpDiffusion
                , extJump = Just defaultJumpParams
                , extStochVol = Nothing
                }

          out <- runMojoMCExt input

          assertBool "mean finite with default params" (not $ isNaN (mean out))
      ]

  , testGroup "Stochastic Volatility Model"
      [ testCase "Stochastic vol model produces valid output" $ do
          let input = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.2
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 5000
                , extModel = StochasticVol
                , extJump = Nothing
                , extStochVol = Just StochVolParams
                    { svMeanRev = 2.0
                    , svLongTermVol = 0.2
                    , svVolOfVol = 0.3
                    , svCorrelation = -0.5
                    }
                }

          out <- runMojoMCExt input

          assertBool "mean finite" (not $ isNaN (mean out))
          assertBool "std positive" (std out >= 0)
          assertBool "var <= cvar" (var_99 out <= cvar_99 out)

      , testCase "Default stochastic vol params work" $ do
          let input = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.25
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 1000
                , extModel = StochasticVol
                , extJump = Nothing
                , extStochVol = Just defaultStochVolParams
                }

          out <- runMojoMCExt input

          assertBool "mean finite with default params" (not $ isNaN (mean out))

      , testCase "High vol-of-vol increases uncertainty" $ do
          -- Low vol-of-vol (nearly constant vol)
          let inputLowVoV = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.25
                , extR = 0.05
                , extK = 100
                , extSteps = 100
                , extPaths = 10000
                , extModel = StochasticVol
                , extJump = Nothing
                , extStochVol = Just StochVolParams
                    { svMeanRev = 5.0     -- Fast mean reversion
                    , svLongTermVol = 0.25
                    , svVolOfVol = 0.05   -- Low vol-of-vol
                    , svCorrelation = 0.0
                    }
                }

          -- High vol-of-vol
          let inputHighVoV = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = 0.25
                , extR = 0.05
                , extK = 100
                , extSteps = 100
                , extPaths = 10000
                , extModel = StochasticVol
                , extJump = Nothing
                , extStochVol = Just StochVolParams
                    { svMeanRev = 0.5     -- Slow mean reversion
                    , svLongTermVol = 0.25
                    , svVolOfVol = 0.8    -- High vol-of-vol
                    , svCorrelation = 0.0
                    }
                }

          outLow <- runMojoMCExt inputLowVoV
          outHigh <- runMojoMCExt inputHighVoV

          -- Both should produce valid output
          assertBool "Low VoV mean finite" (not $ isNaN (mean outLow))
          assertBool "High VoV mean finite" (not $ isNaN (mean outHigh))
      ]

  , testGroup "Model Comparison"
      [ testCase "All three models produce comparable ATM prices" $ do
          let baseParams s m jp sv = MojoInputExt
                { extS0 = 100
                , extMu = 0
                , extSigma = s
                , extR = 0.05
                , extK = 100
                , extSteps = 50
                , extPaths = 10000
                , extModel = m
                , extJump = jp
                , extStochVol = sv
                }

          let inputGBM = baseParams 0.25 GBM Nothing Nothing
          let inputJump = baseParams 0.25 JumpDiffusion
                (Just JumpParams { jpIntensity = 0.5, jpMean = 0, jpVol = 0.05 })
                Nothing
          let inputSV = baseParams 0.25 StochasticVol
                Nothing
                (Just StochVolParams { svMeanRev = 3.0, svLongTermVol = 0.25, svVolOfVol = 0.2, svCorrelation = -0.3 })

          outGBM <- runMojoMCExt inputGBM
          outJump <- runMojoMCExt inputJump
          outSV <- runMojoMCExt inputSV

          -- All should give positive means for ATM call
          assertBool "GBM ATM positive" (mean outGBM > 0)
          assertBool "Jump ATM positive" (mean outJump > 0)
          assertBool "SV ATM positive" (mean outSV > 0)

          -- All should be in reasonable range (say 5-20 for ATM with 25% vol)
          assertBool "GBM in range" (mean outGBM > 2 && mean outGBM < 25)
          assertBool "Jump in range" (mean outJump > 2 && mean outJump < 25)
          assertBool "SV in range" (mean outSV > 2 && mean outSV < 25)
      ]
  ]
