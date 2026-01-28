{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module MojoBridge
  ( -- * Core types
    MojoInput(..)
  , MojoOutput(..)
  , ModelType(..)
    -- * Extended input types
  , MojoInputExt(..)
  , JumpParams(..)
  , StochVolParams(..)
    -- * Running Monte Carlo
  , runMojoMC
  , runMojoMCExt
    -- * Default parameters
  , defaultJumpParams
  , defaultStochVolParams
  ) where

import System.Process (callCommand)
import Data.Aeson
import Data.Aeson.Types (Pair)
import qualified Data.ByteString.Lazy as B
import GHC.Generics (Generic)

-- ============================================================
-- Model Types
-- ============================================================

-- | Monte Carlo model type
data ModelType = GBM | JumpDiffusion | StochasticVol
  deriving (Show, Eq)

modelToString :: ModelType -> String
modelToString GBM = "gbm"
modelToString JumpDiffusion = "jump"
modelToString StochasticVol = "stochvol"

-- ============================================================
-- Basic Input/Output (backward compatible)
-- ============================================================

-- | Basic input for GBM Monte Carlo
data MojoInput = MojoInput
  { s0    :: Double
  , mu    :: Double
  , sigma :: Double
  , r     :: Double
  , k     :: Double
  , steps :: Int
  , paths :: Int
  } deriving (Show, Generic)

instance ToJSON MojoInput

-- | Output from Monte Carlo simulation
data MojoOutput = MojoOutput
  { var_99  :: Double
  , cvar_99 :: Double
  , mean    :: Double
  , std     :: Double
  } deriving (Show, Generic)

instance FromJSON MojoOutput

-- ============================================================
-- Extended Input for Jump-Diffusion and Stochastic Vol
-- ============================================================

-- | Jump-diffusion (Merton model) parameters
data JumpParams = JumpParams
  { jpIntensity :: Double  -- ^ Expected jumps per year (lambda)
  , jpMean      :: Double  -- ^ Mean of log jump size
  , jpVol       :: Double  -- ^ Volatility of log jump size
  } deriving (Show, Generic)

-- | Default jump parameters (moderate jump risk)
--   ~1 jump per year, -5% mean jump, 10% jump vol
defaultJumpParams :: JumpParams
defaultJumpParams = JumpParams
  { jpIntensity = 1.0
  , jpMean = -0.05
  , jpVol = 0.10
  }

-- | Stochastic volatility (Heston-like) parameters
data StochVolParams = StochVolParams
  { svMeanRev     :: Double  -- ^ Mean reversion speed (kappa)
  , svLongTermVol :: Double  -- ^ Long-term volatility (theta)
  , svVolOfVol    :: Double  -- ^ Volatility of volatility (xi)
  , svCorrelation :: Double  -- ^ Correlation between price and vol (rho)
  } deriving (Show, Generic)

-- | Default stochastic vol parameters (typical equity-like)
defaultStochVolParams :: StochVolParams
defaultStochVolParams = StochVolParams
  { svMeanRev = 2.0       -- Moderate mean reversion
  , svLongTermVol = 0.25  -- 25% long-term vol
  , svVolOfVol = 0.3      -- 30% vol-of-vol
  , svCorrelation = -0.7  -- Negative correlation (leverage effect)
  }

-- | Extended input supporting all model types
data MojoInputExt = MojoInputExt
  { extS0        :: Double
  , extMu        :: Double
  , extSigma     :: Double
  , extR         :: Double
  , extK         :: Double
  , extSteps     :: Int
  , extPaths     :: Int
  , extModel     :: ModelType
  , extJump      :: Maybe JumpParams
  , extStochVol  :: Maybe StochVolParams
  } deriving (Show)

instance ToJSON MojoInputExt where
  toJSON inp = object $ baseFields ++ modelFields ++ jumpFields ++ svFields
    where
      baseFields :: [Pair]
      baseFields =
        [ "s0"    .= extS0 inp
        , "mu"    .= extMu inp
        , "sigma" .= extSigma inp
        , "r"     .= extR inp
        , "k"     .= extK inp
        , "steps" .= extSteps inp
        , "paths" .= extPaths inp
        ]

      modelFields :: [Pair]
      modelFields = [ "model" .= modelToString (extModel inp) ]

      jumpFields :: [Pair]
      jumpFields = case extJump inp of
        Nothing -> []
        Just jp ->
          [ "jump_intensity" .= jpIntensity jp
          , "jump_mean"      .= jpMean jp
          , "jump_vol"       .= jpVol jp
          ]

      svFields :: [Pair]
      svFields = case extStochVol inp of
        Nothing -> []
        Just sv ->
          [ "vol_mean_rev"    .= svMeanRev sv
          , "vol_long_term"   .= svLongTermVol sv
          , "vol_of_vol"      .= svVolOfVol sv
          , "vol_correlation" .= svCorrelation sv
          ]

-- ============================================================
-- Running Monte Carlo
-- ============================================================

-- | Run basic GBM Monte Carlo (backward compatible)
runMojoMC :: MojoInput -> IO MojoOutput
runMojoMC input = do
  B.writeFile "input.json" (encode input)
  callCommand "mojo run mojo/bridge.mojo"
  out <- B.readFile "output.json"
  case decode out of
    Just o  -> pure o
    Nothing -> error $ "Failed to decode Mojo output: " ++ show out

-- | Run extended Monte Carlo with model selection
runMojoMCExt :: MojoInputExt -> IO MojoOutput
runMojoMCExt input = do
  B.writeFile "input.json" (encode input)
  callCommand "mojo run mojo/bridge.mojo"
  out <- B.readFile "output.json"
  case decode out of
    Just o  -> pure o
    Nothing -> error $ "Failed to decode Mojo output: " ++ show out
