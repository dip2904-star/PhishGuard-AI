import pickle
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

# ==================== FEATURE EXTRACTION (Same as training) ====================
def extract_features_single(url):
    """
    Extract features from a single URL (same logic as training)
    """
    features = {}
    
    # 1. Basic URL character features
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_questionmarks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_at'] = url.count('@')
    features['num_ampersand'] = url.count('&')
    features['num_percent'] = url.count('%')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['num_special'] = sum(not c.isalnum() for c in url)
    
    # 2. URL component features
    try:
        parsed = urlparse(url)
        
        if HAS_TLDEXTRACT:
            ext = tldextract.extract(url)
            domain = ext.domain
            subdomain = ext.subdomain
            tld = ext.suffix
        else:
            netloc = parsed.netloc
            parts = netloc.split('.')
            domain = parts[-2] if len(parts) >= 2 else netloc
            subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
            tld = parts[-1] if len(parts) >= 1 else ''
        
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['domain_length'] = len(domain)
        features['subdomain_length'] = len(subdomain)
        features['path_length'] = len(parsed.path)
        features['query_length'] = len(parsed.query)
        features['has_subdomain'] = 1 if len(subdomain) > 0 else 0
        features['num_subdomains'] = subdomain.count('.') + 1 if subdomain else 0
        features['tld_length'] = len(tld)
    except:
        features['has_https'] = 0
        features['domain_length'] = 0
        features['subdomain_length'] = 0
        features['path_length'] = 0
        features['query_length'] = 0
        features['has_subdomain'] = 0
        features['num_subdomains'] = 0
        features['tld_length'] = 0
    
    # 3. Suspicious patterns
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    features['has_ip'] = 1 if ip_pattern.search(url) else 0
    
    suspicious_words = [
        'login', 'signin', 'bank', 'account', 'update', 'verify', 'secure',
        'webscr', 'ebayisapi', 'password', 'credential', 'paypal', 'wallet',
        'confirm', 'suspended', 'urgent', 'alert', 'click', 'here'
    ]
    features['num_suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())
    
    brand_names = ['paypal', 'amazon', 'facebook', 'google', 'microsoft', 'apple', 'netflix']
    features['has_brand_name'] = 1 if any(brand in url.lower() for brand in brand_names) else 0
    
    # 4. Entropy
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
    
    features['url_entropy'] = calculate_entropy(url)
    
    try:
        features['domain_entropy'] = calculate_entropy(domain)
    except:
        features['domain_entropy'] = 0
    
    # 5. Ratio features
    features['digit_ratio'] = features['num_digits'] / features['url_length'] if features['url_length'] > 0 else 0
    features['letter_ratio'] = features['num_letters'] / features['url_length'] if features['url_length'] > 0 else 0
    features['special_ratio'] = features['num_special'] / features['url_length'] if features['url_length'] > 0 else 0
    
    # 6. Additional heuristics
    features['is_shortened'] = 1 if any(short in url.lower() for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
    features['has_double_slash'] = 1 if '//' in url[8:] else 0
    features['abnormal_url'] = int(features['has_ip'] or (features['num_dots'] > 5))
    
    return features

# ==================== LOAD MODEL ====================
def load_model(model_path, scaler_path=None):
    """Load trained model and scaler"""
    print("=" * 60)
    print("LOADING PHISHING DETECTION MODEL")
    print("=" * 60)
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[+] Model loaded: {model_path}")
        
        scaler = None
        if scaler_path:
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"[+] Scaler loaded: {scaler_path}")
            except FileNotFoundError:
                print(f"[!] Scaler not found (not needed for tree-based models)")
        
        return model, scaler
    
    except FileNotFoundError:
        print(f"[-] ERROR: Model file not found: {model_path}")
        print("[!] Make sure you've trained the model first using phishing_model_training.py")
        return None, None

# ==================== PREDICT ====================
def predict_url(url, model, scaler=None):
    """Predict if a URL is phishing or legitimate"""
    
    # Normalize URL
    url = url.strip().lower()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Extract features
    features = extract_features_single(url)
    
    # Convert to DataFrame with correct column order
    # CRITICAL: Must match the exact feature order from training
    feature_order = [
        'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 
        'num_slashes', 'num_questionmarks', 'num_equals', 'num_at', 
        'num_ampersand', 'num_percent', 'num_digits', 'num_letters', 
        'num_special', 'has_https', 'domain_length', 'subdomain_length', 
        'path_length', 'query_length', 'has_subdomain', 'num_subdomains', 
        'tld_length', 'has_ip', 'num_suspicious_words', 'has_brand_name', 
        'url_entropy', 'domain_entropy', 'digit_ratio', 'letter_ratio', 
        'special_ratio', 'is_shortened', 'has_double_slash', 'abnormal_url'
    ]
    
    # Create DataFrame with correct feature order
    features_df = pd.DataFrame([features], columns=feature_order)
    
    # Fill any missing values
    features_df = features_df.fillna(-1)
    
    # Scale if scaler is provided (for Logistic Regression)
    if scaler:
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = proba[prediction]
        else:
            confidence = None
    else:
        prediction = model.predict(features_df)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            confidence = proba[prediction]
        else:
            confidence = None
    
    # HOTFIX: Override for known legitimate domains with short URLs
    # This addresses the training data bias toward long-path URLs
    known_legitimate = [
        'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 
        'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
        'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com',
        'reddit.com', 'netflix.com', 'ebay.com', 'walmart.com'
    ]
    
    # Check if it's a short URL from a known domain
    if features['path_length'] <= 5 and features['url_length'] < 30:
        for domain in known_legitimate:
            if domain in url:
                # Override to legitimate if it's a major brand homepage
                prediction = 0
                confidence = 0.95  # High confidence for whitelist
                features['_whitelist_override'] = True
                break
    
    return prediction, confidence, features

# ==================== DISPLAY RESULT ====================
def display_prediction(url, prediction, confidence, features):
    """Display prediction results in a user-friendly format"""
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"URL: {url}")
    print("-" * 60)
    
    whitelist_override = features.get('_whitelist_override', False)
    
    if prediction == 1:
        print("⚠️  PHISHING DETECTED!")
        print("Status: 🔴 DANGEROUS - DO NOT VISIT")
    else:
        print("✓ LEGITIMATE")
        print("Status: 🟢 SAFE")
        if whitelist_override:
            print("Note: Known legitimate domain (whitelist)")
    
    if confidence:
        print(f"Confidence: {confidence*100:.2f}%")
    
    print("\n" + "-" * 60)
    print("KEY FEATURES DETECTED:")
    print("-" * 60)
    
    if whitelist_override:
        print("  ✓ Recognized major brand domain")
    
    # Show important suspicious features
    if features['has_ip']:
        print("  ⚠️  Contains IP address")
    if features['num_suspicious_words'] > 0:
        print(f"  ⚠️  Contains {features['num_suspicious_words']} suspicious keyword(s)")
    if features['has_brand_name'] and not whitelist_override:
        print("  ⚠️  Contains brand name (possible spoofing)")
    if features['is_shortened']:
        print("  ⚠️  Uses URL shortener")
    if not features['has_https']:
        print("  ⚠️  Not using HTTPS")
    if features['has_double_slash']:
        print("  ⚠️  Contains double slash in path")
    if features['url_length'] > 100:
        print(f"  ⚠️  Unusually long URL ({features['url_length']} chars)")
    if features['num_subdomains'] > 2:
        print(f"  ⚠️  Multiple subdomains ({features['num_subdomains']})")
    
    print("=" * 60)

# ==================== MAIN ====================
def main():
    """Interactive URL phishing detection"""
    print("\n" + "=" * 70)
    print("🔍 PHISHING URL DETECTOR")
    print("=" * 70)
    
    # Find and load model
    import os
    import glob
    
    # Look for model files
    model_files = glob.glob('phishing_detection_model_*.pkl')
    
    if not model_files:
        print("\n[-] ERROR: No trained model found!")
        print("[!] Please run 'phishing_model_training.py' first to train a model")
        return
    
    # Use the first model found
    model_path = model_files[0]
    print(f"[*] Found model: {model_path}")
    scaler_path = 'feature_scaler.pkl' if os.path.exists('feature_scaler.pkl') else None
    
    model, scaler = load_model(model_path, scaler_path)
    
    if model is None:
        return
    
    print("\n[+] Model ready! You can now check URLs for phishing.")
    print("[*] Type 'quit' or 'exit' to stop\n")
    
    # Interactive loop
    while True:
        url = input("Enter URL to check: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("\n[*] Goodbye!")
            break
        
        if not url:
            print("[-] Please enter a valid URL\n")
            continue
        
        try:
            prediction, confidence, features = predict_url(url, model, scaler)
            display_prediction(url, prediction, confidence, features)
            print()
        
        except Exception as e:
            print(f"\n[-] Error processing URL: {str(e)}\n")

# ==================== BATCH PREDICTION ====================
def predict_from_file(input_file, output_file, model, scaler=None):
    """Predict multiple URLs from a CSV file"""
    print(f"\n[*] Reading URLs from: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Detect URL column
    url_cols = [col for col in df.columns if 'url' in col.lower()]
    if not url_cols:
        url_col = df.columns[0]
    else:
        url_col = url_cols[0]
    
    print(f"[*] Using column: {url_col}")
    print(f"[*] Processing {len(df)} URLs...")
    
    predictions = []
    confidences = []
    
    for url in df[url_col]:
        try:
            pred, conf, _ = predict_url(str(url), model, scaler)
            predictions.append(pred)
            confidences.append(conf if conf else 0)
        except:
            predictions.append(-1)  # Error
            confidences.append(0)
    
    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].map({0: 'Legitimate', 1: 'Phishing', -1: 'Error'})
    df['Confidence'] = confidences
    
    df.to_csv(output_file, index=False)
    print(f"\n[+] Results saved to: {output_file}")
    print(f"    - Phishing: {(df['Prediction']==1).sum()}")
    print(f"    - Legitimate: {(df['Prediction']==0).sum()}")
    print(f"    - Errors: {(df['Prediction']==-1).sum()}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Batch mode: python predict.py input.csv output.csv
        if len(sys.argv) >= 3:
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            
            model_files = glob.glob('phishing_detection_model_*.pkl')
            if model_files:
                print(f"[*] Using model: {model_files[0]}")
                model, scaler = load_model(model_files[0], 'feature_scaler.pkl')
                if model:
                    predict_from_file(input_file, output_file, model, scaler)
        else:
            print("Usage: python predict.py input.csv output.csv")
    else:
        # Interactive mode
        main()