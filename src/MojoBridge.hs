{-# LANGUAGE DeriveGeneric #-}

module MojoBridge where

import System.Process (callCommand)
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import GHC.Generics (Generic)

-- Keep the same data types as tests expect
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

data MojoOutput = MojoOutput
  { var_99  :: Double
  , cvar_99 :: Double
  , mean    :: Double
  , std     :: Double
  } deriving (Show, Generic)

instance FromJSON MojoOutput

-- Writes input.json, runs Mojo, reads output.json
runMojoMC :: MojoInput -> IO MojoOutput
runMojoMC input = do
  B.writeFile "input.json" (encode input)
  -- invoke the vectorized Mojo engine
  -- use the same filename mojo/bridge.mojo so "mojo run mojo/bridge.mojo" works
  callCommand "mojo run mojo/bridge.mojo"
  out <- B.readFile "output.json"
  case decode out of
    Just o  -> pure o
    Nothing -> error $ "Failed to decode Mojo output: " ++ show out

