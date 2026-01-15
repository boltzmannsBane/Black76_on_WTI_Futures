{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Map.Strict as M
import Data.Time (Day, defaultTimeLocale, parseTimeM, diffDays)
import qualified Data.Vector as V
import Text.Printf (printf)
import System.Environment (getArgs)

import Black76

-- Futures CSV expected to have: Date (MM/DD/YYYY), Price
data FutRecord = FutRecord { fDate :: Day, fPrice :: Maybe Double }

instance FromNamedRecord FutRecord where
  parseNamedRecord r = do
    dateStr <- r .: "Date"
    -- parse MM/DD/YYYY
    day <- case (parseTimeM True defaultTimeLocale "%m/%d/%Y" (dateStr :: String) :: Maybe Day) of
             Just d  -> pure d
             Nothing -> fail $ "Invalid futures date: " ++ dateStr
    price <- optional (r .: "Price")
    pure $ FutRecord day price

-- time to maturity (years)
timeToMaturity :: Day -> Day -> Double
timeToMaturity current expire = max 0 (fromIntegral (diffDays expire current) / 365.0)

-- Simple main: read futures CSV, price Black-76 call + Greeks (w.r.t forward)
-- Usage: wti-black76-exe /path/to/wti_futures.csv YYYY-MM-DD (expiry)
main :: IO ()
main = do
    args <- getArgs
    case args of
      [futPath, expiryStr] -> do
        futRaw <- BL.readFile futPath
        let futMap = case decodeByName futRaw of
              Left err -> error ("Futures CSV parse error: " ++ err)
              Right (_, v) ->
                V.foldl' (\acc (FutRecord d p) ->
                            case p of
                              Just price -> M.insert d price acc
                              Nothing    -> acc
                         ) M.empty v

        -- parse expiry date (YYYY-MM-DD)
        expiry <- case (parseTimeM True defaultTimeLocale "%Y-%m-%d" (expiryStr :: String) :: Maybe Day) of
                    Just d  -> pure d
                    Nothing -> error $ "Invalid expiry date: " ++ expiryStr

        let results = M.mapWithKey (\d f ->
                        let t = timeToMaturity d expiry
                            price = black76Call f 75.0 t 0.04 0.35  -- K/r/sigma here are example defaults
                            deltaF = black76DeltaF f 75.0 t 0.04 0.35
                            gammaF = black76GammaF f 75.0 t 0.04 0.35
                        in (f, t, price, deltaF, gammaF)
                      ) futMap

        putStrLn "Date | Future | T (yrs) | B76_Call | Delta_F | Gamma_F"
        mapM_ (\(d, (f, t, p, df, gf)) ->
                putStrLn $ show d ++ " | "
                          ++ printf "%.2f" f ++ " | "
                          ++ printf "%.6f" t ++ " | "
                          ++ printf "%.6f" p ++ " | "
                          ++ printf "%.6f" df ++ " | "
                          ++ printf "%.8f" gf
              ) (reverse $ M.toList results)

      _ -> putStrLn "Usage: wti-black76-exe /path/to/wti_futures.csv YYYY-MM-DD"
