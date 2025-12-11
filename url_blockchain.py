import hashlib
import json
import time
from typing import List, Dict


class Block:
    def __init__(self, index: int, timestamp: float, urls: List[Dict], previous_hash: str):
        """
        Initialize a block in the blockchain
        
        Args:
            index: Block number in the chain
            timestamp: Time of block creation
            urls: List of URL dictionaries containing url, status, and confidence
            previous_hash: Hash of the previous block
        """
        self.index = index
        self.timestamp = timestamp
        self.urls = urls
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "urls": self.urls,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class URLBlockchain:
    def __init__(self):
        """Initialize the blockchain with a genesis block"""
        self.chain = [self.create_genesis_block()]
        self.pending_urls = []
        self.block_size = 10  # Number of URLs per block
        self.url_cache = {}  # Cache for quick URL lookups
        self.reported_urls = set()  # Set to track reported phishing URLs

    def create_genesis_block(self) -> Block:
        """Create the first block in the chain"""
        return Block(0, time.time(), [], "0")

    def get_last_block(self) -> Block:
        """Return the most recent block in the chain"""
        return self.chain[-1]

    def add_url(self, url: str, status: str, confidence: float, is_reported: bool = False, phishing_reports: int = None) -> bool:
        """
        Add a URL to pending transactions
        
        Args:
            url: The URL to store
            status: Classification status (phishing/legitimate/pending)
            confidence: Model's confidence score
            is_reported: Whether this URL was reported by a user
            phishing_reports: Number of phishing reports for this URL
        
        Returns:
            bool: True if block was created, False if URL was added to pending
        """
        print(f"Adding URL: {url}, Status: {status}, Is Reported: {is_reported}")  # Debug log
        current_time = time.time()
        
        # If URL was previously reported as phishing, maintain that status
        if url in self.reported_urls:
            status = 'reported_phishing'
            confidence = 1.0
            is_reported = True
            print("URL was previously reported as phishing")  # Debug log
        
        # Get existing data if URL exists
        existing_data = self.search_url(url)
        print(f"Existing data found: {existing_data}")  # Debug log
        
        url_data = {
            "url": url,
            "status": status,
            "confidence": confidence,
            "timestamp": current_time,
            "last_checked": current_time,
            "is_reported": is_reported,
            "phishing_reports": phishing_reports if phishing_reports is not None else (1 if is_reported else 0)
        }

        # If URL exists, preserve or update certain fields
        if existing_data:
            url_data["phishing_reports"] = existing_data.get("phishing_reports", 0)
            if is_reported:
                url_data["phishing_reports"] += 1
            if existing_data.get("is_reported", False):
                url_data["is_reported"] = True
        
        if is_reported:
            self.reported_urls.add(url)
            url_data["report_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update cache
        self.url_cache[url] = url_data
        
        # Add to pending URLs
        self.pending_urls.append(url_data)

        if len(self.pending_urls) >= self.block_size:
            return self.mine_pending_urls()
        return False

    def mine_pending_urls(self) -> bool:
        """
        Create a new block with pending URLs
        
        Returns:
            bool: True if block was created successfully
        """
        if not self.pending_urls:
            return False

        last_block = self.get_last_block()
        new_block = Block(
            last_block.index + 1,
            time.time(),
            self.pending_urls[:self.block_size],
            last_block.hash
        )

        # Add the new block to the chain
        self.chain.append(new_block)
        # Remove the URLs that were just added to the block
        self.pending_urls = self.pending_urls[self.block_size:]
        return True

    def is_chain_valid(self) -> bool:
        """
        Verify the integrity of the blockchain
        
        Returns:
            bool: True if the chain is valid
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Verify current block's hash
            if current_block.hash != current_block.calculate_hash():
                return False

            # Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_all_urls(self) -> List[Dict]:
        """
        Get all URLs stored in the blockchain
        
        Returns:
            List[Dict]: List of all URLs and their details
        """
        all_urls = []
        for block in self.chain:
            all_urls.extend(block.urls)
        return all_urls

    def search_url(self, url: str) -> Dict:
        """
        Search for a specific URL in the blockchain and cache
        
        Args:
            url: URL to search for
        
        Returns:
            Dict: URL details if found, None otherwise
        """
        # First check the cache
        if url in self.url_cache:
            return self.url_cache[url]
        
        # Then check pending URLs
        for url_data in self.pending_urls:
            if url_data["url"] == url:
                self.url_cache[url] = url_data
                return url_data
        
        # Finally check the blockchain
        for block in self.chain:
            for url_data in block.urls:
                if url_data["url"] == url:
                    self.url_cache[url] = url_data
                    return url_data
        
        return None

    def export_chain(self, filename: str):
        """
        Export the blockchain to a JSON file
        
        Args:
            filename: Name of the file to save the blockchain
        """
        chain_data = []
        for block in self.chain:
            chain_data.append({
                "index": block.index,
                "timestamp": block.timestamp,
                "urls": block.urls,
                "hash": block.hash,
                "previous_hash": block.previous_hash
            })
        
        with open(filename, 'w') as f:
            json.dump(chain_data, f, indent=4)

    def import_chain(self, filename: str):
        """
        Import blockchain from a JSON file
        
        Args:
            filename: Name of the file to load the blockchain from
        """
        with open(filename, 'r') as f:
            chain_data = json.load(f)
        
        self.chain = []
        for block_data in chain_data:
            block = Block(
                block_data["index"],
                block_data["timestamp"],
                block_data["urls"],
                block_data["previous_hash"]
            )
            block.hash = block_data["hash"]
            self.chain.append(block)


# Example usage
if __name__ == "__main__":
    # Create blockchain instance
    blockchain = URLBlockchain()

    # Add some example URLs
    test_urls = [
        ("http://example.com", "legitimate", 0.95),
        ("http://suspicious-site.com", "phishing", 0.88),
        ("http://safe-site.com", "legitimate", 0.92)
    ]

    # Add URLs to blockchain
    for url, status, confidence in test_urls:
        blockchain.add_url(url, status, confidence)

    # Force mining of any remaining URLs
    if blockchain.pending_urls:
        blockchain.mine_pending_urls()

    # Verify chain integrity
    print("Blockchain valid:", blockchain.is_chain_valid())

    # Export chain to file
    blockchain.export_chain("url_chain.json")

    # Search for a URL
    result = blockchain.search_url("http://example.com")
    if result:
        print(f"Found URL: {result}")