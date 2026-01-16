import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    print("Warning: tldextract not installed. Install with: pip install tldextract")

# ==================== SMART DATASET LOADER ====================
class UniversalPhishingDatasetLoader:
    """
    Automatically detects and loads any phishing dataset format
    """
    
    # Common column name patterns
    URL_PATTERNS = ['url', 'uri', 'link', 'website', 'domain', 'address', 'site']
    LABEL_PATTERNS = ['label', 'class', 'type', 'category', 'status', 'result', 'target']
    
    # Common label values for phishing
    PHISHING_VALUES = ['phishing', 'bad', 'malicious', 'fraud', 'scam', '1', 1, 'positive', 'suspicious', 'malware', 'defacement']
    LEGITIMATE_VALUES = ['legitimate', 'good', 'benign', 'safe', 'normal', '0', 0, 'negative', 'valid']
    
    def __init__(self):
        self.url_column = None
        self.label_column = None
        self.dataset_format = "unknown"
    
    def detect_columns(self, df):
        """
        Intelligently detect URL and Label columns
        """
        print("\n[*] Auto-detecting columns...")
        columns_lower = {col: col.lower().strip() for col in df.columns}
        
        # Detect URL column
        for col, col_lower in columns_lower.items():
            # Check by name
            if any(pattern in col_lower for pattern in self.URL_PATTERNS):
                self.url_column = col
                print(f"[+] URL column detected: '{col}'")
                break
            
        # If not found by name, check by content (look for http/www patterns)
        if self.url_column is None:
            for col in df.columns:
                sample = df[col].dropna().astype(str).head(10)
                if sample.str.contains(r'http|www|\.com|\.org|\.net', case=False, regex=True).sum() >= 5:
                    self.url_column = col
                    print(f"[+] URL column detected by content: '{col}'")
                    break
        
        # Detect Label column
        for col, col_lower in columns_lower.items():
            if col == self.url_column:
                continue
            
            # Check by name
            if any(pattern in col_lower for pattern in self.LABEL_PATTERNS):
                self.label_column = col
                print(f"[+] Label column detected: '{col}'")
                break
        
        # If not found by name, look for binary/categorical column
        if self.label_column is None:
            for col in df.columns:
                if col == self.url_column:
                    continue
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 5:  # Binary or few categories
                    self.label_column = col
                    print(f"[+] Label column detected by uniqueness: '{col}'")
                    break
        
        if self.url_column is None:
            raise ValueError("[-] Could not detect URL column! Please specify manually.")
        if self.label_column is None:
            raise ValueError("[-] Could not detect Label column! Please specify manually.")
        
        return self.url_column, self.label_column
    
    def standardize_labels(self, df):
        """
        Convert any label format to binary: 1=phishing, 0=legitimate
        """
        print("\n[*] Standardizing labels...")
        label_col = 'Label_Original'  # Use the renamed column
        
        # Get unique values
        unique_labels = df[label_col].unique()
        print(f"Original labels found: {unique_labels}")
        
        # Convert to string for comparison
        df[label_col] = df[label_col].astype(str).str.lower().str.strip()
        
        # Create binary labels
        def classify_label(x):
            x = str(x).lower().strip()
            
            # Check if phishing
            for phish in self.PHISHING_VALUES:
                if str(phish).lower() in x:
                    return 1
            
            # Check if legitimate
            for legit in self.LEGITIMATE_VALUES:
                if str(legit).lower() in x:
                    return 0
            
            # Default: if numeric
            try:
                val = float(x)
                return 1 if val > 0 else 0
            except:
                # Unknown - mark as phishing to be safe
                return 1
        
        df['Label'] = df[label_col].apply(classify_label)
        
        print(f"[+] Labels standardized:")
        print(f"    - Phishing (1): {(df['Label']==1).sum()}")
        print(f"    - Legitimate (0): {(df['Label']==0).sum()}")
        
        return df
    
    def load(self, file_path, url_col=None, label_col=None, sep=',', encoding='utf-8'):
        """
        Load and standardize any phishing dataset
        
        Parameters:
        - file_path: path to CSV file
        - url_col: manually specify URL column (optional)
        - label_col: manually specify label column (optional)
        - sep: CSV separator (default: ',')
        - encoding: file encoding (default: 'utf-8')
        """
        print("=" * 60)
        print("UNIVERSAL PHISHING DATASET LOADER")
        print("=" * 60)
        
        # Try different encodings if default fails
        encodings = [encoding, 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=enc, on_bad_lines='skip')
                print(f"[+] File loaded successfully with encoding: {enc}")
                break
            except Exception as e:
                if enc == encodings[-1]:
                    raise Exception(f"[-] Could not load file with any encoding: {e}")
                continue
        
        print(f"[*] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[*] Columns: {list(df.columns)}")
        
        # Manual or auto-detect columns
        if url_col and label_col:
            self.url_column = url_col
            self.label_column = label_col
            print(f"\n[+] Using manually specified columns:")
            print(f"    - URL: '{url_col}'")
            print(f"    - Label: '{label_col}'")
        else:
            self.detect_columns(df)
        
        # Rename to standard names
        df = df.rename(columns={
            self.url_column: 'URL',
            self.label_column: 'Label_Original'
        })
        
        # Standardize labels
        df = self.standardize_labels(df)
        
        # Keep only necessary columns
        df = df[['URL', 'Label']]
        
        print(f"\n[+] Dataset standardized successfully!")
        return df

# ==================== 2. DATA CLEANING ====================
def clean_data(df):
    """
    Remove duplicates, nulls, invalid URLs and normalize formats
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)
    
    initial_count = len(df)
    print(f"Initial records: {initial_count}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['URL'], keep='first')
    print(f"After removing duplicates: {len(df)} ({initial_count - len(df)} removed)")
    
    # Remove null/empty URLs
    df = df.dropna(subset=['URL'])
    df = df[df['URL'].astype(str).str.strip() != '']
    print(f"After removing null/empty URLs: {len(df)}")
    
    # Convert to string and normalize
    df['URL'] = df['URL'].astype(str).str.strip()
    df['URL'] = df['URL'].str.lower()
    
    # Remove common prefixes/suffixes that might cause issues
    df['URL'] = df['URL'].str.replace(r'^[\'"]+|[\'"]+$', '', regex=True)
    
    # Add protocol if missing
    def add_protocol(url):
        if not url.startswith(('http://', 'https://', 'ftp://')):
            return 'http://' + url
        return url
    
    df['URL'] = df['URL'].apply(add_protocol)
    
    # Basic URL validation (regex-based, faster than validators library)
    def is_valid_url(url):
        try:
            # Basic URL pattern
            pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
                r'localhost|'  # localhost
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            return pattern.match(url) is not None
        except:
            return False
    
    df['is_valid'] = df['URL'].apply(is_valid_url)
    invalid_count = (~df['is_valid']).sum()
    df = df[df['is_valid']].drop('is_valid', axis=1)
    print(f"After removing invalid URLs: {len(df)} ({invalid_count} removed)")
    
    # Remove URLs that are too short (likely invalid)
    df = df[df['URL'].str.len() >= 10]
    print(f"After removing too short URLs: {len(df)}")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"\n[+] Final clean dataset: {len(df)} records")
    print(f"    - Phishing: {(df['Label']==1).sum()}")
    print(f"    - Legitimate: {(df['Label']==0).sum()}")
    
    return df

# ==================== 3. FEATURE EXTRACTION ====================
def extract_features(df, max_samples=None):
    """
    Extract comprehensive features from URLs
    """
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE EXTRACTION")
    print("=" * 60)
    
    # Limit samples if specified (for faster processing)
    if max_samples and len(df) > max_samples:
        print(f"[!] Limiting to {max_samples} samples for faster processing")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    features_df = pd.DataFrame()
    features_df['URL'] = df['URL'].values
    
    # 1. Basic URL character features
    print("[*] Extracting basic URL features...")
    features_df['url_length'] = df['URL'].apply(len)
    features_df['num_dots'] = df['URL'].str.count(r'\.')
    features_df['num_hyphens'] = df['URL'].str.count('-')
    features_df['num_underscores'] = df['URL'].str.count('_')
    features_df['num_slashes'] = df['URL'].str.count('/')
    features_df['num_questionmarks'] = df['URL'].str.count(r'\?')
    features_df['num_equals'] = df['URL'].str.count('=')
    features_df['num_at'] = df['URL'].str.count('@')
    features_df['num_ampersand'] = df['URL'].str.count('&')
    features_df['num_percent'] = df['URL'].str.count('%')
    features_df['num_digits'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in x))
    features_df['num_letters'] = df['URL'].apply(lambda x: sum(c.isalpha() for c in x))
    features_df['num_special'] = df['URL'].apply(lambda x: sum(not c.isalnum() for c in x))
    
    # 2. URL component features
    print("[*] Parsing URL components...")
    
    def safe_parse_url(url):
        try:
            parsed = urlparse(url)
            
            # Extract domain parts
            if HAS_TLDEXTRACT:
                ext = tldextract.extract(url)
                domain = ext.domain
                subdomain = ext.subdomain
                tld = ext.suffix
            else:
                # Fallback parsing
                netloc = parsed.netloc
                parts = netloc.split('.')
                domain = parts[-2] if len(parts) >= 2 else netloc
                subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
                tld = parts[-1] if len(parts) >= 1 else ''
            
            return {
                'scheme': parsed.scheme,
                'domain': domain,
                'subdomain': subdomain,
                'tld': tld,
                'path': parsed.path,
                'query': parsed.query,
                'fragment': parsed.fragment,
                'netloc': parsed.netloc
            }
        except:
            return {
                'scheme': '', 'domain': '', 'subdomain': '', 'tld': '',
                'path': '', 'query': '', 'fragment': '', 'netloc': ''
            }
    
    parsed_urls = df['URL'].apply(safe_parse_url)
    
    features_df['has_https'] = parsed_urls.apply(lambda x: 1 if x['scheme'] == 'https' else 0)
    features_df['domain_length'] = parsed_urls.apply(lambda x: len(x['domain']))
    features_df['subdomain_length'] = parsed_urls.apply(lambda x: len(x['subdomain']))
    features_df['path_length'] = parsed_urls.apply(lambda x: len(x['path']))
    features_df['query_length'] = parsed_urls.apply(lambda x: len(x['query']))
    features_df['has_subdomain'] = parsed_urls.apply(lambda x: 1 if len(x['subdomain']) > 0 else 0)
    features_df['num_subdomains'] = parsed_urls.apply(lambda x: x['subdomain'].count('.') + 1 if x['subdomain'] else 0)
    features_df['tld_length'] = parsed_urls.apply(lambda x: len(x['tld']))
    
    # 3. Suspicious patterns
    print("[*] Detecting suspicious patterns...")
    
    # IP address in URL
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    features_df['has_ip'] = df['URL'].apply(lambda x: 1 if ip_pattern.search(x) else 0)
    
    # Suspicious keywords
    suspicious_words = [
        'login', 'signin', 'bank', 'account', 'update', 'verify', 'secure',
        'webscr', 'ebayisapi', 'password', 'credential', 'paypal', 'wallet',
        'confirm', 'suspended', 'urgent', 'alert', 'click', 'here'
    ]
    features_df['num_suspicious_words'] = df['URL'].apply(
        lambda x: sum(1 for word in suspicious_words if word in x.lower())
    )
    
    # Brand name spoofing check
    brand_names = ['paypal', 'amazon', 'facebook', 'google', 'microsoft', 'apple', 'netflix']
    features_df['has_brand_name'] = df['URL'].apply(
        lambda x: 1 if any(brand in x.lower() for brand in brand_names) else 0
    )
    
    # 4. Entropy and randomness
    print("[*] Computing statistical features...")
    
    def calculate_entropy(text):
        if len(text) == 0:
            return 0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0
        for count in freq.values():
            p = count / len(text)
            entropy -= p * np.log2(p)
        return entropy
    
    features_df['url_entropy'] = df['URL'].apply(calculate_entropy)
    features_df['domain_entropy'] = parsed_urls.apply(lambda x: calculate_entropy(x['domain']))
    
    # 5. Ratio features
    features_df['digit_ratio'] = features_df['num_digits'] / features_df['url_length']
    features_df['letter_ratio'] = features_df['num_letters'] / features_df['url_length']
    features_df['special_ratio'] = features_df['num_special'] / features_df['url_length']
    
    # 6. Additional heuristics
    features_df['is_shortened'] = df['URL'].apply(
        lambda x: 1 if any(short in x.lower() for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
    )
    features_df['has_double_slash'] = df['URL'].apply(lambda x: 1 if '//' in x[8:] else 0)
    features_df['abnormal_url'] = (features_df['has_ip'] | (features_df['num_dots'] > 5))
    
    # Add label
    features_df['Label'] = df['Label'].values
    
    print(f"\n[+] Extracted {len(features_df.columns)-2} features")
    print(f"[*] Feature names: {[col for col in features_df.columns if col not in ['URL', 'Label']]}")
    
    return features_df

# ==================== 4. DATASET SPLITTING ====================
def split_and_save_dataset(features_df, output_dir='./', train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset and save to files
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATASET SPLITTING & SAVING")
    print("=" * 60)
    
    # Shuffle dataset
    features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Handle missing values
    missing = features_df.isnull().sum()
    if missing.sum() > 0:
        print(f"[!] Filling {missing.sum()} missing values with -1")
        features_df = features_df.fillna(-1)
    
    # Check class balance
    print("\n[*] Class distribution:")
    print(f"    - Phishing (1): {(features_df['Label']==1).sum()}")
    print(f"    - Legitimate (0): {(features_df['Label']==0).sum()}")
    balance = (features_df['Label']==1).sum() / (features_df['Label']==0).sum()
    print(f"    - Balance ratio: {balance:.2f}")
    
    if balance < 0.5 or balance > 2.0:
        print("    [!] Dataset is imbalanced - consider using class weights in training")
    
    # Split dataset
    n = len(features_df)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    train_df = features_df[:train_size]
    val_df = features_df[train_size:train_size + val_size]
    test_df = features_df[train_size + val_size:]
    
    print(f"\n[*] Dataset splits:")
    print(f"    - Train: {len(train_df)} ({len(train_df)/n*100:.1f}%) | Phishing: {(train_df['Label']==1).sum()}, Legitimate: {(train_df['Label']==0).sum()}")
    print(f"    - Val: {len(val_df)} ({len(val_df)/n*100:.1f}%) | Phishing: {(val_df['Label']==1).sum()}, Legitimate: {(val_df['Label']==0).sum()}")
    print(f"    - Test: {len(test_df)} ({len(test_df)/n*100:.1f}%) | Phishing: {(test_df['Label']==1).sum()}, Legitimate: {(test_df['Label']==0).sum()}")
    
    # Save to CSV
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n[+] Datasets saved successfully:")
    print(f"    - {train_path}")
    print(f"    - {val_path}")
    print(f"    - {test_path}")
    
    return train_df, val_df, test_df

# ==================== MAIN PIPELINE ====================
def main(input_file, output_dir='./', url_col=None, label_col=None, 
         max_samples=None, sep=',', encoding='utf-8'):
    """
    Universal phishing dataset processing pipeline
    
    Parameters:
    - input_file: path to dataset CSV
    - output_dir: directory to save processed datasets
    - url_col: URL column name (auto-detected if None)
    - label_col: Label column name (auto-detected if None)
    - max_samples: limit number of samples for faster processing
    - sep: CSV separator
    - encoding: file encoding
    """
    print("\n" + "=" * 70)
    print("UNIVERSAL PHISHING WEBSITE DETECTION PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Load data with universal loader
        loader = UniversalPhishingDatasetLoader()
        df = loader.load(input_file, url_col, label_col, sep, encoding)
        
        # Step 2: Clean data
        df_clean = clean_data(df)
        
        # Step 3: Extract features
        features_df = extract_features(df_clean, max_samples)
        
        # Step 4: Split and save
        train_df, val_df, test_df = split_and_save_dataset(features_df, output_dir)
        
        print("\n" + "=" * 70)
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n[*] You can now use these datasets for machine learning!")
        print(f"[*] Total features: {len(train_df.columns) - 2}")  # Exclude URL and Label
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"\n[-] ERROR: {str(e)}")
        print("\n[!] Troubleshooting tips:")
        print("    1. Check if file path is correct")
        print("    2. Try specifying url_col and label_col manually")
        print("    3. Check if file encoding is correct (try 'latin-1' or 'utf-8')")
        print("    4. Ensure file is in CSV format")
        raise

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("[*] RUNNING PIPELINE...")
    print("=" * 70)
    
    # Run the actual pipeline
    input_file = "phishing_site_urls.csv" 
    output_dir = "./"
    
    train_df, val_df, test_df = main(
        input_file, 
        output_dir=output_dir,
        max_samples=None  # Set to 10000 for quick testing
    )