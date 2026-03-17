#!/usr/bin/env python3
"""
Debug script to verify aggregation query returns only correct destination ports
for 192.168.0.17
"""
import json
from core.runner import Runner
from core.config import Config

def debug_aggregation():
    cfg = Config()
    runner = Runner(cfg)
    db = runner.db
    
    target_ip = "192.168.0.17"
    
    # Build the exact same aggregation query used in ip_fingerprinter
    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": "now-30d", "format": "epoch_millis"}}},
                ],
                "should": [
                    {"term": {"dest_ip": target_ip}},
                    {"term": {"destination.ip": target_ip}},
                ],
                "minimum_should_match": 1,
            }
        },
        "aggs": {
            "dest_ports": {
                "terms": {
                    "field": "dest_port",
                    "size": 100,
                }
            }
        },
        "size": 0,
    }
    
    print(f"[DEBUG] Aggregation query for {target_ip}:")
    print(json.dumps(query, indent=2))
    print()
    
    try:
        aggs = db.aggregate("logstash*", query)
        dest_buckets = aggs.get("dest_ports", {}).get("buckets", [])
        
        print(f"[DEBUG] Aggregation returned {len(dest_buckets)} ports:")
        for bucket in dest_buckets:
            port = bucket.get("key")
            count = bucket.get("doc_count", 0)
            print(f"  Port {port}: {count} records")
        
        # Now verify each port by checking individual records
        print()
        print(f"[DEBUG] Verifying each port with sample records:")
        for bucket in dest_buckets[:5]:  # Check first 5 ports
            port = bucket.get("key")
            verify_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"bool": {
                                "should": [
                                    {"term": {"dest_ip": target_ip}},
                                    {"term": {"destination.ip": target_ip}},
                                ],
                                "minimum_should_match": 1,
                            }},
                            {"bool": {
                                "should": [
                                    {"term": {"dest_port": port}},
                                    {"term": {"destination.port": port}},
                                ],
                                "minimum_should_match": 1,
                            }},
                        ]
                    }
                }
            }
            
            results = db.search("logstash*", verify_query, size=2)
            print(f"\n  Port {port} - sample records:")
            for i, record in enumerate(results[:2]):
                dest_ip = record.get("dest_ip") or record.get("destination", {}).get("ip")
                dest_port = record.get("dest_port") or record.get("destination", {}).get("port")
                src_ip = record.get("src_ip") or record.get("source", {}).get("ip")
                src_port = record.get("src_port") or record.get("source", {}).get("port")
                print(f"    [{i+1}] src={src_ip}:{src_port} -> dest={dest_ip}:{dest_port}")
    
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    debug_aggregation()
