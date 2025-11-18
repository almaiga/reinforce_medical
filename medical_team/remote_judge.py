"""
Remote Medical Judge Client

Client for calling the medical judge HTTP endpoint.
Compatible with OpenRLHF's remote reward model interface.
"""

import requests
from typing import Dict, List, Any
import time


def create_medical_judge_remote_function(judge_url: str, timeout: int = 30):
    """
    Create a remote judge function that calls the HTTP endpoint.
    
    Args:
        judge_url: Base URL of the judge server (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds
        
    Returns:
        Function that takes (url, batch_queries, score_key) and returns results
    """
    
    def remote_judge_fn(url: str, batch_queries: List[Dict[str, Any]], score_key: str = "labels"):
        """
        Call remote judge endpoint with batch queries.
        
        Args:
            url: Ignored (uses judge_url from closure)
            batch_queries: List of query dicts with game_idx, medical_note, etc.
            score_key: Key for scores (default "labels")
            
        Returns:
            Dict mapping game_idx to labels dict
        """
        endpoint = f"{judge_url}/judge"
        
        payload = {
            "queries": batch_queries,
            "score_key": score_key
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Convert string keys to int keys if needed
            if results and isinstance(list(results.keys())[0], str):
                results = {int(k): v for k, v in results.items()}
            
            return results
            
        except requests.exceptions.Timeout:
            print(f"❌ Judge request timed out after {timeout}s")
            # Return safe defaults for all queries
            return {
                q["game_idx"]: {
                    "error_detected": False,
                    "error_present": False,
                    "error_realistic": True,
                    "assessor_correct": False,
                    "is_parsing_error": True,
                    "judge_reasoning": "Timeout",
                    "actual_harm": "Safe"
                }
                for q in batch_queries
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Judge request failed: {e}")
            # Return safe defaults for all queries
            return {
                q["game_idx"]: {
                    "error_detected": False,
                    "error_present": False,
                    "error_realistic": True,
                    "assessor_correct": False,
                    "is_parsing_error": True,
                    "judge_reasoning": f"Error: {str(e)}",
                    "actual_harm": "Safe"
                }
                for q in batch_queries
            }
    
    return remote_judge_fn


def test_judge_connection(judge_url: str) -> bool:
    """
    Test connection to judge server.
    
    Args:
        judge_url: Base URL of the judge server
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = requests.get(f"{judge_url}/health", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Judge server healthy: {data}")
        return True
        
    except Exception as e:
        print(f"❌ Judge server connection failed: {e}")
        return False


def wait_for_judge_server(judge_url: str, max_wait: int = 60, check_interval: int = 2):
    """
    Wait for judge server to become available.
    
    Args:
        judge_url: Base URL of the judge server
        max_wait: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        True if server becomes available, False if timeout
    """
    print(f"⏳ Waiting for judge server at {judge_url}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if test_judge_connection(judge_url):
            return True
        
        time.sleep(check_interval)
    
    print(f"❌ Judge server did not become available within {max_wait}s")
    return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test medical judge connection")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Judge server URL"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for server to become available"
    )
    
    args = parser.parse_args()
    
    if args.wait:
        success = wait_for_judge_server(args.url)
        exit(0 if success else 1)
    else:
        success = test_judge_connection(args.url)
        exit(0 if success else 1)
