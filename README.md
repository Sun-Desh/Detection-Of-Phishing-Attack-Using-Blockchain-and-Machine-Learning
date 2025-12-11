# URL Phishing Detection using BERT and Blockchain

## Detailed Technical Documentation

### Understanding the Metrics

#### 1. Confusion Matrix Components
In binary classification (phishing vs. legitimate):
```
┌─────────────────────────────────────────────────────┐
│              Actual                                 │
│           P        N                               │
│   ┌────────┬────────┐                             │
│ P │   TP   │   FP   │                             │
│   │        │        │  P = Phishing               │
│   ├────────┼────────┤  N = Legitimate             │
│ N │   FN   │   TN   │  TP = True Positive        │
│   │        │        │  TN = True Negative         │
│   └────────┴────────┘  FP = False Positive        │
│     Predicted          FN = False Negative        │
└─────────────────────────────────────────────────────┘
```

#### 2. Key Performance Metrics

##### 2.1 Accuracy
- **What it is**: Overall correctness of all predictions
- **Formula**: $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- **In our code**:
```python
acc = accuracy_score(labels, preds)
```
- **Interpretation**: Percentage of correct predictions (both phishing and legitimate)
- **When to use**: When classes are balanced (equal number of phishing and legitimate URLs)

##### 2.2 Precision
- **What it is**: Accuracy of positive predictions
- **Formula**: $Precision = \frac{TP}{TP + FP}$
- **In our code**:
```python
precision = precision_score(labels, preds, average='binary')
```
- **Interpretation**: Of the URLs we labeled as phishing, how many were actually phishing
- **When to use**: When false positives are costly (blocking legitimate URLs)

##### 2.3 Recall (Sensitivity)
- **What it is**: Ability to find all positive cases
- **Formula**: $Recall = \frac{TP}{TP + FN}$
- **In our code**:
```python
recall = recall_score(labels, preds, average='binary')
```
- **Interpretation**: Of all actual phishing URLs, how many did we detect
- **When to use**: When false negatives are costly (missing phishing URLs)

##### 2.4 F1 Score
- **What it is**: Harmonic mean of precision and recall
- **Formula**: $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **In our code**:
```python
f1 = f1_score(labels, preds, average='binary')
```
- **Interpretation**: Balance between precision and recall
- **When to use**: When you need a single metric that balances false positives and negatives

##### 2.5 Specificity
- **What it is**: True negative rate
- **Formula**: $Specificity = \frac{TN}{TN + FP}$
- **In our code**:
```python
tn = ((preds == 0) & (labels == 0)).sum()
fp = ((preds == 1) & (labels == 0)).sum()
specificity = tn / (tn + fp)
```
- **Interpretation**: Of all legitimate URLs, how many did we correctly identify
- **When to use**: When measuring the model's ability to identify legitimate URLs

## Project Overview

This project implements an advanced phishing URL detection system combining state-of-the-art machine learning (BERT) with blockchain technology for result persistence. The system provides real-time URL analysis, user reporting capabilities, and maintains an immutable record of detected phishing attempts.

## Table of Contents
1. [Architecture](#architecture)
2. [Components](#components)
3. [Technologies Used](#technologies-used)
4. [Implementation Details](#implementation-details)
5. [Algorithms & Models](#algorithms--models)
6. [Benefits & Use Cases](#benefits--use-cases)
7. [Limitations](#limitations)
8. [Future Enhancements](#future-enhancements)
9. [Setup & Installation](#setup--installation)
10. [Results & Performance](#results--performance)

## Architecture

### High-Level System Design
```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Flask Frontend │────▶│  BERT Model  │────▶│   Blockchain   │
└─────────────────┘     └──────────────┘     └────────────────┘
        ▲                      │                      │
        │                      ▼                      ▼
        │               ┌──────────────┐     ┌────────────────┐
        └───────────────│ URL Analysis │     │  Result Cache  │
                        └──────────────┘     └────────────────┘
```

## Components

### 1. Machine Learning Model (`bert.ipynb`)
- Uses BERT (Bidirectional Encoder Representations from Transformers)
- Fine-tuned for URL classification
- Key metrics tracked:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - Specificity

#### Model Architecture
- Base Model: bert-base-uncased
- Additional Layers:
  - Dropout (0.2)
  - Binary Classification Head
- Training Parameters:
  - Batch Size: 32
  - Learning Rate: 5e-5
  - Epochs: 4
  - Optimizer: AdamW
  - Loss Function: CrossEntropyLoss with class weights

### 2. Blockchain Implementation (`url_blockchain.py`)
- Custom blockchain for URL storage
- Features:
  - SHA-256 hashing
  - Immutable record keeping
  - Quick URL lookup cache
  - Permanent phishing reports storage

#### Block Structure
```python
{
    "index": int,
    "timestamp": float,
    "urls": List[Dict],
    "previous_hash": str,
    "hash": str
}
```

### 3. Web Interface (`app.py` & `templates/`)
- Flask-based web application
- Features:
  - URL submission
  - Real-time analysis
  - Phishing reporting
  - Result history access

## Technologies Used

1. **Machine Learning**:
   - PyTorch
   - Transformers
   - scikit-learn
   - NumPy
   - Pandas

2. **Web Development**:
   - Flask
   - HTML/CSS
   - JavaScript

3. **Storage & Processing**:
   - Custom Blockchain
   - JSON
   - Python's hashlib

## Implementation Details

### Data Preprocessing
1. URL Cleaning:
   ```python
   def preprocess_url(url):
       url = url.lower().strip()
       if not url.startswith(('http://', 'https://')):
           url = 'http://www.' + url.replace('www.', '')
       return url
   ```

2. Feature Extraction:
   ```python
   def extract_url_features(url):
       suspicious_patterns = ['login', 'secure', 'account', ...]
       suspicious_count = sum(1 for pattern in suspicious_patterns if pattern in url)
       return suspicious_count
   ```

### Model Training Process
1. Data Loading & Splitting:
   - 50/50 train-test split
   - Stratified sampling for balanced classes

2. Tokenization:
   - Maximum length: 256 tokens
   - Special tokens: [CLS], [SEP]
   - WordPiece tokenization

3. Training Configuration:
   ```python
   training_args = TrainingArguments(
       learning_rate=5e-5,
       weight_decay=0.01,
       warmup_ratio=0.1,
       lr_scheduler_type='cosine_with_restarts'
   )
   ```

### Blockchain Integration
1. Block Creation:
   - Every 10 URLs form a new block
   - Each block contains:
     - URLs
     - Analysis results
     - Timestamps
     - Previous block's hash

2. URL Status Tracking:
   ```python
   url_data = {
       "url": processed_url,
       "status": prediction,
       "confidence": confidence,
       "timestamp": current_time,
       "is_reported": reported_by_user
   }
   ```

## Algorithms & Models

### BERT Model Technical Details

#### 1. Model Architecture
```
┌─────────────────────────────────────────────┐
│               BERT Architecture             │
├─────────────────────────────────────────────┤
│ Input Layer (WordPiece Tokenization)        │
├─────────────────────────────────────────────┤
│ Embedding Layer                             │
│  - Token Embeddings                         │
│  - Position Embeddings                      │
│  - Segment Embeddings                       │
├─────────────────────────────────────────────┤
│ 12 Transformer Blocks                       │
│  Each block contains:                       │
│  - Multi-head Self-attention (12 heads)     │
│  - Feed Forward Neural Network              │
│  - Layer Normalization                      │
│  - Residual Connections                     │
├─────────────────────────────────────────────┤
│ Pooler Layer                               │
├─────────────────────────────────────────────┤
│ Classification Head                         │
└─────────────────────────────────────────────┘
```

#### 2. Model Parameters
- **Base Architecture**: bert-base-uncased
- **Hidden Size**: 768 dimensions
- **Attention Heads**: 12
- **Layers**: 12 transformer blocks
- **Total Parameters**: 110M
- **Vocabulary Size**: 30,522 tokens

#### 3. Training Process

##### 3.1 Loss Function
Cross-Entropy Loss with class weights:

$L = -\sum_{i=1}^{C} w_i y_i \log(\hat{y_i})$

where:
- $C$ is number of classes (2 for binary)
- $w_i$ is the class weight
- $y_i$ is true label
- $\hat{y_i}$ is predicted probability

##### 3.2 Learning Rate Schedule
Cosine schedule with warmup:

$\text{LR}(t) = \text{LR}_{\text{base}} \cdot \left(\frac{1}{2} \left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}} \pi\right)\right)\right)$

where:
- $t$ is current step
- $t_{\text{warmup}}$ is warmup steps
- $t_{\text{total}}$ is total steps

##### 3.3 Attention Mechanism
Multi-head attention for token $i$:

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

where:
- $Q$ is query matrix
- $K$ is key matrix
- $V$ is value matrix
- $d_k$ is attention head dimension

#### 4. Fine-tuning Configuration
```python
training_args = TrainingArguments(
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=4
)
```

#### 5. Model Output Processing
1. **Raw Logits**: Model outputs unnormalized scores
2. **Softmax**: Convert to probabilities
   $P(class_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$
3. **Confidence**: Highest probability
4. **Decision Threshold**: 0.8 for high confidence

### URL Analysis and Feature Detection

#### 1. URL Structure Analysis
```
┌─────────────────────────────────────────────────────────────┐
│ https://  subdomain.  domain  .com  /path  ?query  #fragment│
│    │          │         │      │      │      │        │     │
│ Protocol   Sub-    Main    TLD   Path  Query  Fragment      │
│            Domain  Domain        Component                   │
└─────────────────────────────────────────────────────────────┘
```

#### 2. Feature Extraction Process

##### 2.1 Lexical Features
- **URL Length**: Longer URLs are more suspicious
  ```python
  length_score = min(len(url) / 100, 1.0)
  ```
- **Domain Length**: Unusual domain lengths indicate suspicion
  ```python
  domain_length = len(domain_part)
  ```
- **Number of Dots**: More dots suggest subdomain manipulation
  ```python
  dots_count = url.count('.')
  ```

##### 2.2 Character Distribution
- **Special Characters**: Presence of @, -, _, %, etc.
  ```python
  special_char_ratio = len([c for c in url if not c.isalnum()]) / len(url)
  ```
- **Number Distribution**: Ratio of numbers in URL
  ```python
  number_ratio = len([c for c in url if c.isdigit()]) / len(url)
  ```

##### 2.3 Pattern Analysis
```python
suspicious_patterns = [
    'login', 'signin', 'verify', 'secure', 'account',
    'update', 'confirm', 'banking', 'paypal', 'password'
]
pattern_score = sum(1 for pattern in suspicious_patterns if pattern in url.lower())
```

##### 2.4 Domain Analysis
- **TLD Checking**: Comparing against common TLDs
- **Domain Age**: If available through WHOIS
- **SSL Certificate**: HTTPS presence and validity

#### 3. BERT Tokenization Process
```python
def tokenize_url(url):
    # WordPiece tokenization
    tokens = tokenizer(
        url,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_attention_mask=True
    )
    return tokens
```

#### 4. Feature Importance Calculation
For each feature $f_i$, importance score $I_i$ is calculated as:

$I_i = \sum_{j=1}^{N} |w_{ij}|$

where $w_{ij}$ is the weight of feature $i$ in decision $j$

#### 5. Confidence Score Calculation
Final confidence score is computed as:

$Confidence = \sigma(BERT_{score} + \alpha \cdot Pattern_{score} + \beta \cdot Structure_{score})$

where:
- $\sigma$ is the sigmoid function
- $\alpha, \beta$ are weight parameters
- $BERT_{score}$ is the model's raw prediction
- $Pattern_{score}$ is from suspicious pattern matching
- $Structure_{score}$ is from URL structure analysis

### 3. Confidence Calculation
```python
confidence = softmax(model_output)[predicted_class]
if suspicious_patterns >= 2:
    confidence = min(confidence + 0.1, 1.0)
```

## Benefits & Use Cases

### 1. End Users
- Real-time phishing protection
- Community-driven reporting
- Historical URL checking

### Advanced Analysis Features

#### 1. Model Performance Analysis

##### 1.1 ROC Curve Analysis
The Receiver Operating Characteristic (ROC) curve plots True Positive Rate vs False Positive Rate:

$TPR = \frac{TP}{TP + FN}$ vs $FPR = \frac{FP}{FP + TN}$

Area Under Curve (AUC) calculation:
$AUC = \int_0^1 TPR(FPR^{-1}(x))dx$

##### 1.2 Precision-Recall Curve
Plots Precision vs Recall at different thresholds:
```python
precision_recall_curve(y_true, y_pred_proba)
```

##### 1.3 Confusion Matrix Analysis
Detailed breakdown of predictions:
```
              Predicted
           P         N
Actual P │   TP    │    FN    │
       N │   FP    │    TN    │
```

#### 2. Security Analysis Features

##### 2.1 Pattern Detection
- **Suspicious Word Detection**:
  ```python
  def detect_suspicious_words(url):
      score = sum(weight * (pattern in url.lower())
                 for pattern, weight in SUSPICIOUS_PATTERNS.items())
      return normalize_score(score)
  ```

- **Character Distribution Analysis**:
  ```python
  def analyze_char_distribution(url):
      char_freq = Counter(url)
      entropy = -sum(p * math.log2(p) 
                    for p in [count/len(url) for count in char_freq.values()])
      return entropy
  ```

##### 2.2 Domain Analysis
```python
def analyze_domain(url):
    parts = extract_domain_parts(url)
    return {
        'domain_length': len(parts['domain']),
        'subdomain_count': len(parts['subdomains']),
        'tld_type': classify_tld(parts['tld']),
        'domain_age': get_domain_age(parts['domain'])
    }
```

##### 2.3 URL Structure Analysis
Comprehensive URL parsing:
```python
def parse_url_structure(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }
```

#### 3. Threat Intelligence Features

##### 3.1 Historical Analysis
```python
def analyze_historical_data(url):
    blockchain_data = blockchain.search_url(url)
    return {
        'times_reported': blockchain_data.get('report_count', 0),
        'first_seen': blockchain_data.get('first_seen'),
        'last_status': blockchain_data.get('last_status'),
        'confidence_history': blockchain_data.get('confidence_history', [])
    }
```

##### 3.2 Pattern Evolution
Tracking pattern changes over time:
```python
def track_pattern_evolution():
    patterns = defaultdict(lambda: {'count': 0, 'first_seen': None})
    for block in blockchain.chain:
        for url_data in block.urls:
            update_pattern_statistics(patterns, url_data)
    return analyze_pattern_trends(patterns)
```

##### 3.3 Threat Scoring
Combined threat score calculation:
```python
def calculate_threat_score(url_data):
    weights = {
        'model_confidence': 0.4,
        'pattern_score': 0.2,
        'structure_score': 0.2,
        'historical_score': 0.2
    }
    
    return sum(weight * score_function(url_data)
              for score_type, weight in weights.items())
```

### 3. Organizations
- Employee protection
- Network security
- Fraud prevention

## Limitations

1. **Technical Limitations**
   - URL format variations
   - New phishing techniques
   - Processing speed vs accuracy

2. **Model Limitations**
   - Training data bias
   - False positives/negatives
   - Resource requirements

3. **Blockchain Limitations**
   - Local storage only
   - No consensus mechanism
   - Limited scalability

## Future Enhancements

1. **Model Improvements**
   - Additional features
   - Regular retraining
   - Ensemble methods

2. **Blockchain Upgrades**
   - Distributed network
   - Smart contracts
   - Proof of work/stake

3. **Feature Additions**
   - API access
   - Browser extension
   - Mobile application

## Setup & Installation

```bash
# Clone repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Setup environment
python -m venv env
source env/bin/activate

# Run application
python app.py
```

## Results & Performance

### Model Performance
- Accuracy: ~94%
- F1 Score: ~0.93
- Precision: ~0.92
- Recall: ~0.94
- Specificity: ~0.93

### System Performance
- Average response time: <1 second
- Blockchain query time: <0.1 seconds
- Memory usage: ~500MB

### Dataset Statistics
- Total URLs: 11,430
- Phishing URLs: 5,715
- Legitimate URLs: 5,715

## Conclusion

This project demonstrates the effective combination of modern machine learning techniques with blockchain technology for phishing detection. The system provides a robust, user-friendly solution for URL analysis while maintaining an immutable record of detected threats.

The implementation shows promising results in both accuracy and performance, while the blockchain integration ensures result persistence and enables community-driven reporting for improved detection rates.

---

## Contact & Contributors

[Your Name/Team Information]

## License

[Specify License]