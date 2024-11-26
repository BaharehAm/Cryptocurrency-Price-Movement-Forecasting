from imports import *

class CryptoDataProcessor:
    def __init__(self, df, config_file='config.yaml'):
        """Initialize with a DataFrame."""
        self.df = df
        # Load configuration from the YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.wavelet = config['wavelet'] 
        self.level = config['level']   
        print(f"Wavelet: {self.wavelet}, Level: {self.level}")

        
    def create_ohlc_data(self):
        """Creates the 'OHLC' feature set and calculates log returns and binary labels."""
        print("Initial DataFrame Columns:", self.df.columns)
        self.df = self.df.rename(columns={'Price': 'Close'})  # Rename 'Price' to 'Close'
        # Reverse the DataFrame because the raw data in CSV files are typically ordered from oldest to newest,
        # and we need it to be in chronological order for analysis.
        self.df = self.df[::-1]  
        self.df.drop(columns=['Vol.', 'Change %'], inplace=True) # Drop unnecessary columns
        print("OHLC DataFrame Columns:", self.df.columns)
        
        # Remove commas and convert to float for the next steps
        numeric_columns = ['Close', 'Open', 'High', 'Low']
        self.df[numeric_columns] = self.df[numeric_columns].replace(',', '', regex=True).astype(float) 
        
        self.calculate_log_return_and_labels()
        
        return self.df

    def calculate_log_return_and_labels(self):
        """Calculates log returns and assigns binary labels."""
        # Calculate log returns
        log_ret = np.log(self.df['Close']) - np.log(self.df['Close'].shift(1))
        self.df['return'] = log_ret * 100  # Percentage values
        self.df['return'].fillna(0, inplace=True)

        # Assign binary labels
        self.df['binary_label'] = self.df['return'].apply(lambda x: 1 if x > 0 else 0)

        # Replace infinity values with a small number
        self.df.replace([np.inf, -np.inf], 0.000001, inplace=True)
        
        
    def create_ti_data(self, add_correlated=False, correlation_threshold=0.85, all_data_df=None):
        """Calculates technical indicators using StockDataFrame and adds correlated close prices."""
        col_name = 'Close'
        column = self.df.pop(col_name)  # Remove the column from the DataFrame
        self.df.insert(len(self.df.columns)-2, col_name, column)  # Insert the column at the desired position
        
        # Create a StockDataFrame object
        self.df = Sdf.retype(self.df)

        # Call StockDataFrame methods to calculate technical indicators
        self.df['close_5_sma']  # 5-day simple moving average
        self.df['close_10_sma']  # 10-day simple moving average
        self.df['close_20_sma']  # 20-day simple moving average
        self.df['close_5_ema']  # 5-day Exponential moving average
        self.df['close_10_ema']  # 10-day Exponential moving average
        self.df['close_20_ema']  # 20-day Exponential moving average
        self.df['rsi_14']  # Relative Strength Index (RSI)
        self.df['boll']  # Bollinger Bands
        self.df['boll_ub']  # Bollinger upper band
        self.df['boll_lb']  # Bollinger lower band
        self.df['boll_15']  # Bollinger Bands
        self.df['boll_ub_15']  # Bollinger upper band
        self.df['boll_lb_15']  # Bollinger lower band
        self.df['macd']  # MACD
        self.df['macds']  # MACD signal
        self.df['macdh']  # MACD histogram

        # Stochastic oscillators
        self.df.KDJ_WINDOW = 5
        self.df['kdjk_5']  # K series
        self.df['kdjd_5']  # D series
        self.df.KDJ_WINDOW = 9
        self.df['kdjk_9']  # K series
        self.df['kdjd_9']  # D series
        self.df.KDJ_WINDOW = 14
        self.df['kdjk_14']  # K series
        self.df['kdjd_14']  # D series

        # Williams %R
        self.df['wr_14']  # Williams %R with a period of 14
        self.df['wr_6']  # Williams %R with a period of 6

        if add_correlated and all_data_df is not None:
            self.add_correlated_prices(all_data_df, correlation_threshold)

        self.calculate_log_return_and_labels()
        
        return self.df

    def add_correlated_prices(self, all_data_df, threshold):
        """Add highly correlated close prices to the DataFrame."""
        correlation_matrix = all_data_df.corr()
        crypto_corr = correlation_matrix[self.df['Close'].name]
        highly_correlated = crypto_corr[crypto_corr.abs() > threshold].index.tolist()
        print("Cryptocurrencies with correlations > {:.2f}: {}".format(threshold, highly_correlated))

        for cr in highly_correlated:
            if cr != self.df['Close'].name:
                self.df[cr] = all_data_df[cr]  # Add the correlated close prices

        self.df.drop(columns=[self.df['Close'].name], inplace=True)  # Remove original close prices

    def create_dwt_data(self):
        """Applies DWT to the feature set and removes noise from the features."""
        # DWT implementation can be added here if needed
        for column in self.df.columns[:-2]:  # Exclude return and binary_label
            data = self.df[column].values
            coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

            # Apply thresholding to the wavelet coefficients
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))  # Universal threshold
            coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]

            # Reconstruct the denoised signal using the modified wavelet coefficients
            denoised_data = pywt.waverec(coeffs, self.wavelet)
            
            # Ensure lengths match before assignment
            if len(denoised_data) == len(data):
                self.df[column] = denoised_data
            else:
                print(f"Length mismatch for {column}: Original {len(data)}, Denoised {len(denoised_data)}")
                # Slice denoised_data to match original length
                self.df[column] = denoised_data[:len(data)]  
            self.calculate_log_return_and_labels()
        
        return self.df
