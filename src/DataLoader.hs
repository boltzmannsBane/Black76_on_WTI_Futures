module DataLoader
  ( PricePoint(..)
  , loadWTIPrices
  , pricesToList
  ) where

import Data.List (sortOn)

data PricePoint = PricePoint
  { ppDate  :: String
  , ppPrice :: Double
  } deriving (Show, Eq)

-- | Load WTI futures prices from CSV file
-- Format: Date,Price,Open,High,Low,Vol.,Change %
-- Returns prices in chronological order (oldest first)
loadWTIPrices :: FilePath -> IO [PricePoint]
loadWTIPrices path = do
  content <- readFile path
  let rows = drop 1 $ lines content  -- skip header
      parsed = map parseRow rows
  pure $ reverse parsed  -- CSV is newest-first, we want oldest-first

parseRow :: String -> PricePoint
parseRow line =
  let fields = parseCSVLine line
      date  = stripQuotes (fields !! 0)
      price = read (stripQuotes (fields !! 1)) :: Double
  in PricePoint date price

-- | Parse a CSV line handling quoted fields
parseCSVLine :: String -> [String]
parseCSVLine = go False ""
  where
    go _ acc [] = [reverse acc]
    go inQuote acc (c:cs)
      | c == '"'  = go (not inQuote) acc cs
      | c == ',' && not inQuote = reverse acc : go False "" cs
      | otherwise = go inQuote (c:acc) cs

stripQuotes :: String -> String
stripQuotes s = filter (/= '"') s

-- | Extract just the prices as a list of Doubles
pricesToList :: [PricePoint] -> [Double]
pricesToList = map ppPrice
