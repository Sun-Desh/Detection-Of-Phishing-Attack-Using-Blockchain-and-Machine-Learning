from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from url_blockchain import URLBlockchain
import os
import time

app = Flask(__name__)

# Initialize the blockchain
blockchain = URLBlockchain()

# Load the BERT model and tokenizer
model_path = './phishing_detection_model_new'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = 128

def preprocess_url(url):
    """Preprocess URL for consistent format"""
    url = url.lower().strip()
    
    # Add http:// if no protocol is specified
    if not url.startswith(('http://', 'https://')):
        if url.startswith('www.'):
            url = 'http://' + url
        else:
            url = 'http://www.' + url.replace('www.', '')
    
    return url

def extract_url_features(url):
    """Extract features that might indicate phishing"""
    suspicious_patterns = [
        'login', 'signin', 'verify', 'secure', 'account',
        'update', 'confirm', 'banking', 'paypal', 'password'
    ]
    
    url_lower = url.lower()
    suspicious_count = sum(1 for pattern in suspicious_patterns if pattern in url_lower)
    has_suspicious_chars = '@' in url or '%' in url
    dots_count = url.count('.')
    
    return suspicious_count, has_suspicious_chars, dots_count

def predict_url(url, max_length=None):
    """Enhanced URL prediction with better detection"""
    if max_length is None:
        max_length = min(256, len(url) + 10)
    
    # Preprocess URL
    processed_url = preprocess_url(url)
    
    # Get additional features
    suspicious_count, has_suspicious_chars, dots_count = extract_url_features(processed_url)
    
    inputs = tokenizer(
        processed_url,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    predictions = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(predictions).item()
    confidence = predictions[0][predicted_class].item()
    
    # Adjust confidence based on additional features
    if predicted_class == 1:  # If model predicts phishing
        if suspicious_count >= 2 or (has_suspicious_chars and dots_count > 2):
            confidence = min(confidence + 0.1, 1.0)
    else:  # If model predicts legitimate
        if suspicious_count >= 2 and dots_count > 3:
            confidence = max(confidence - 0.1, 0.0)
            predicted_class = 1  # Change to phishing if highly suspicious
    
    threshold = 0.8
    prediction = 'phishing' if predicted_class == 1 else 'legitimate'
    if confidence < threshold:
        confidence_label = "LOW"
    elif confidence < 0.9:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "HIGH"
    
    return {
        'url': url,
        'prediction': prediction,
        'confidence': f"{confidence:.2%}",
        'confidence_level': confidence_label,
        'sequence_length': max_length
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_url', methods=['POST'])
def check_url():
    url = request.form.get('url')
    if not url:
        return render_template('index.html', error="Please enter a URL")
    
    # Preprocess URL for consistent format
    processed_url = preprocess_url(url)
    
    # Check if this URL was previously reported as phishing
    existing_result = blockchain.search_url(processed_url)
    
    if existing_result and existing_result.get('is_reported', False):
        # If URL was reported as phishing, always return it as phishing
        result = {
            'url': url,
            'processed_url': processed_url,
            'prediction': 'Reported Phishing',
            'confidence': '100%',
            'confidence_level': 'HIGH',
            'from_blockchain': True,
            'reported_by_user': True,
            'report_time': existing_result.get('report_time', 'Unknown'),
            'last_checked': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    elif existing_result and existing_result['status'] != 'pending':
        # URL was previously analyzed but not reported
        result = {
            'url': url,
            'processed_url': processed_url,
            'prediction': existing_result['status'],
            'confidence': f"{existing_result['confidence']:.2%}",
            'confidence_level': 'HIGH' if existing_result['confidence'] > 0.9 else 'MEDIUM' if existing_result['confidence'] > 0.8 else 'LOW',
            'from_blockchain': True,
            'last_checked': existing_result.get('timestamp', 'Unknown')
        }
    else:
        # Store as pending first
        blockchain.add_url(
            url=processed_url,
            status='pending',
            confidence=0.0
        )
        
        # Get new prediction
        result = predict_url(url)
        # Update blockchain with the prediction
        blockchain.add_url(
            url=processed_url,
            status=result['prediction'],
            confidence=float(result['confidence'].strip('%'))/100
        )
        result['from_blockchain'] = False
    
    return render_template('index.html', result=result)

@app.route('/report_url', methods=['POST'])
def report_url():
    try:
        url = request.form.get('url')
        print(f"Received URL report request for: {url}")  # Debug log
        
        if not url:
            print("No URL provided")  # Debug log
            return render_template('index.html', error="Please enter a URL")
        
        # Preprocess URL for consistent format
        processed_url = preprocess_url(url)
        print(f"Processed URL: {processed_url}")  # Debug log
        
        # Check existing status in blockchain
        existing_result = blockchain.search_url(processed_url)
        print(f"Existing blockchain result: {existing_result}")  # Debug log
        
        # First, handle well-known legitimate URLs
        well_known_legitimate = [
            'google.com', 'facebook.com', 'microsoft.com', 'apple.com',
            'amazon.com', 'github.com', 'linkedin.com', 'twitter.com'
        ]
    except Exception as e:
        # Log the error and return a user-friendly message
        print(f"Error processing URL report: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Detailed error log
        return render_template('index.html', error="An error occurred while processing your report. Please try again.")
    domain = processed_url.split('/')[2] if '/' in processed_url else processed_url
    base_domain = '.'.join(domain.split('.')[-2:])  # Get base domain
    
    if base_domain in well_known_legitimate:
        report_data = {
            'url': processed_url,
            'processed_url': processed_url,
            'prediction': 'Verified Legitimate',
            'status': 'verified_legitimate',
            'confidence': '100%',
            'confidence_level': 'HIGH',
            'message': 'This is a verified legitimate URL that cannot be reported as phishing.',
            'report_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        return render_template('index.html', result=report_data)
    
    # For other URLs, check their existing status
    if existing_result:
        # If URL was previously analyzed with high confidence as legitimate
        if (existing_result['status'] == 'legitimate' and 
            existing_result['confidence'] > 0.95 and 
            not existing_result.get('phishing_reports', 0)):
            
            report_data = {
                'url': processed_url,
                'processed_url': processed_url,
                'prediction': 'Likely Legitimate',
                'status': 'legitimate',
                'confidence': f"{existing_result['confidence']:.2%}",
                'confidence_level': 'HIGH',
                'message': 'This URL has been verified as legitimate with high confidence. Multiple reports needed to mark as phishing.',
                'report_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Track the report but don't change status yet
            phishing_reports = existing_result.get('phishing_reports', 0) + 1
            try:
                blockchain.add_url(
                    url=processed_url,
                    status='legitimate',
                    confidence=existing_result['confidence'],
                    phishing_reports=phishing_reports
                )
            except Exception as e:
                print(f"Error updating blockchain: {str(e)}")
                return render_template('index.html', error="Failed to update the URL status. Please try again.")
            
            if phishing_reports >= 3:  # Require multiple reports to override legitimate status
                blockchain.add_url(
                    url=processed_url,
                    status='reported_phishing',
                    confidence=0.8,  # Lower confidence due to conflicting signals
                    is_reported=True,
                    phishing_reports=phishing_reports
                )
                report_data['message'] = 'Multiple reports have flagged this URL. Marking as suspicious.'
                report_data['status'] = 'reported_phishing'
            
            return render_template('index.html', result=report_data)
    
    # For new or uncertain URLs, proceed with regular reporting
    report_data = {
        'url': processed_url,
        'processed_url': processed_url,
        'prediction': 'Reported Phishing',
        'status': 'reported_phishing',
        'confidence': '90%',  # Slightly lower confidence for user reports
        'confidence_level': 'HIGH',
        'reported_by_user': True,
        'report_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add to blockchain with report tracking
    try:
        print("Attempting to add URL to blockchain...")  # Debug log
        
        # Calculate phishing_reports value
        if existing_result:
            current_reports = existing_result.get('phishing_reports', 0)
            print(f"Current phishing reports: {current_reports}")  # Debug log
            new_reports = current_reports + 1
        else:
            new_reports = 1
        
        result = blockchain.add_url(
            url=processed_url,
            status='reported_phishing',
            confidence=0.9,  # Slightly lower confidence for user reports
            is_reported=True,
            phishing_reports=new_reports
        )
        print(f"URL added to blockchain. Result: {result}")  # Debug log
        
        # Force mine any pending URLs
        if blockchain.pending_urls:
            print("Mining pending URLs...")  # Debug log
            blockchain.mine_pending_urls()
        
        # Export the updated blockchain
        blockchain.export_chain('url_chain.json')
        
    except Exception as e:
        print(f"Error adding URL to blockchain: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Detailed error log
        return render_template('index.html', error="Failed to save your report. Please try again.")
    
    return render_template('index.html', result=report_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Load existing blockchain if available
    if os.path.exists('url_chain.json'):
        blockchain.import_chain('url_chain.json')
    
    app.run(debug=True, port=5000)