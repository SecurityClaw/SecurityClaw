"""
tests/test_threat_analyst_context_enrichment.py

Tests that verify supervisor passes previous results to threat_analyst
so it can analyze SPECIFIC entities rather than generic questions.

This fixes the bug where:
- opensearch_querier found 2 Iran IPs
- threat_analyst was called but didn't know which IPs to analyze
- Result: threat_analyst returned generic reputation info instead of analyzing the discovered IPs
"""

import pytest
from skills.chat_router.logic import (
    _extract_entities_from_previous_results,
    _build_context_aware_threat_question,
)


class TestEntityExtraction:
    """Test extraction of IPs, domains, countries from previous results."""

    def test_extract_ips_from_opensearch_results(self):
        """Verify IPs are extracted from opensearch_querier results."""
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 2,
                "results": [
                    {"src_ip": "1.2.3.4", "dst_ip": "5.6.7.8"},
                    {"src_ip": "9.10.11.12", "dst_ip": "13.14.15.16"},
                ],
                "countries": ["Iran"],
                "ports": ["443", "80"],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        assert "1.2.3.4" in entities["ips"]
        assert "9.10.11.12" in entities["ips"]
        assert "Iran" in entities["countries"]
        assert "443" in entities["ports"]
        assert "opensearch_querier" in entities["sources"]

    def test_extract_domains_from_opensearch_results(self):
        """Verify domains are extracted from opensearch_querier results."""
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "results": [
                    {"src_ip": "1.2.3.4", "domain": "example.com"},
                    {"src_ip": "5.6.7.8", "hostname": "server.malicious.org"},
                ],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        assert "1.2.3.4" in entities["ips"]
        assert "example.com" in entities["domains"]
        assert "server.malicious.org" in entities["domains"]

    def test_extract_from_baseline_querier_results(self):
        """Verify IPs and ports are extracted from baseline_querier results."""
        aggregated = {
            "baseline_querier": {
                "status": "ok",
                "ips": ["192.168.1.1", "10.0.0.1"],
                "ports": ["8080", "3389"],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        assert "192.168.1.1" in entities["ips"]
        assert "10.0.0.1" in entities["ips"]
        assert "8080" in entities["ports"]
        assert "3389" in entities["ports"]
        assert "baseline_querier" in entities["sources"]

    def test_extract_from_both_sources(self):
        """Verify extraction works when both opensearch and rag querier have results."""
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "results": [
                    {"src_ip": "1.2.3.4", "country": "Iran"},
                ],
                "countries": ["Iran"],
                "ports": ["443"],
            },
            "baseline_querier": {
                "status": "ok",
                "ips": ["9.10.11.12"],
                "ports": ["80"],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        assert "1.2.3.4" in entities["ips"]
        assert "9.10.11.12" in entities["ips"]
        assert "Iran" in entities["countries"]
        assert "443" in entities["ports"]
        assert "80" in entities["ports"]
        assert len(entities["sources"]) == 2

    def test_empty_results_returns_empty_entities(self):
        """Verify empty results return empty entity dict."""
        aggregated = {}
        entities = _extract_entities_from_previous_results(aggregated)

        assert entities["ips"] == []
        assert entities["domains"] == []
        assert entities["countries"] == []
        assert entities["ports"] == []
        assert entities["sources"] == []

    def test_extract_handles_malformed_data(self):
        """Verify extraction gracefully handles malformed data."""
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "results": [
                    {"src_ip": None},  # None value
                    {"src_ip": 12345},  # Numeric instead of string
                    {"src_ip": "1.2.3.4"},  # Valid
                    {"results": "not a list but string"},  # Entire field is wrong
                ],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        # Should extract only the valid IP
        assert "1.2.3.4" in entities["ips"]
        assert len(entities["ips"]) == 1

    def test_extract_does_not_trust_requested_country_metadata_on_validation_failure(self):
        """Validation-failed OpenSearch results should not inject requested countries as discovered facts."""
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "validation_failed": True,
                "results": [
                    {"src_ip": "192.168.0.85", "dst_ip": "92.63.103.84"},
                ],
                "countries": ["Russia"],
                "ports": ["443"],
            }
        }

        entities = _extract_entities_from_previous_results(aggregated)

        assert "192.168.0.85" in entities["ips"]
        assert "92.63.103.84" in entities["ips"]
        assert entities["countries"] == []


class TestContextAwareQuestionEnrichment:
    """Test that threat_analyst questions are enriched with discovered entities."""

    def test_enrich_question_with_single_ip(self):
        """Verify question is enriched with a single discovered IP."""
        original_q = "what's the reputation?"
        entities = {
            "ips": ["1.2.3.4"],
            "domains": [],
            "countries": [],
            "ports": [],
        }

        enriched = _build_context_aware_threat_question(original_q, entities)

        assert "1.2.3.4" in enriched
        assert "Previously discovered entities" in enriched
        assert original_q in enriched

    def test_enrich_question_with_multiple_ips(self):
        """Verify question is enriched with multiple IPs (limited to 5)."""
        original_q = "what's the threat level?"
        ips = [f"1.2.3.{i}" for i in range(10)]
        entities = {
            "ips": ips,
            "domains": [],
            "countries": [],
            "ports": [],
        }

        enriched = _build_context_aware_threat_question(original_q, entities)

        # Should include first 5 IPs
        assert "1.2.3.0" in enriched
        assert "1.2.3.4" in enriched
        # Should indicate more IPs exist
        assert "(and more)" in enriched
        assert original_q in enriched

    def test_enrich_question_with_countries_and_ports(self):
        """Verify question is enriched with countries and ports."""
        original_q = "what is the reputation?"
        entities = {
            "ips": ["1.2.3.4"],
            "domains": [],
            "countries": ["Iran", "Russia"],
            "ports": ["443", "80", "8080"],
        }

        enriched = _build_context_aware_threat_question(original_q, entities)

        assert "1.2.3.4" in enriched
        assert "Iran" in enriched
        assert "Russia" in enriched
        assert "443" in enriched
        assert original_q in enriched

    def test_no_enrichment_if_no_entities(self):
        """Verify question unchanged if no entities discovered."""
        original_q = "what's the reputation?"
        entities = {
            "ips": [],
            "domains": [],
            "countries": [],
            "ports": [],
        }

        enriched = _build_context_aware_threat_question(original_q, entities)

        # Should return original question unchanged
        assert enriched == original_q

    def test_enrich_with_domains(self):
        """Verify question is enriched with discovered domains."""
        original_q = "what is the reputation?"
        entities = {
            "ips": [],
            "domains": ["malware.com", "c2server.ru"],
            "countries": [],
            "ports": [],
        }

        enriched = _build_context_aware_threat_question(original_q, entities)

        assert "malware.com" in enriched
        assert "c2server.ru" in enriched
        assert "Previously discovered entities" in enriched


class TestIntegrationScenario:
    """Integration test mimicking the actual bug scenario."""

    def test_iran_traffic_follow_up_scenario(self):
        """
        Simulate: User asks "traffic from iran in the past 3 years?"
        - opensearch_querier finds 2 records with specific IPs
        - supervisor calls threat_analyst to analyze those IPs
        - threat_analyst should receive enriched question with actual IPs
        """
        # Simulate results from opensearch_querier's first pass
        aggregated = {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 2,
                "results": [
                    {
                        "src_ip": "185.220.101.52",
                        "dst_ip": "203.0.113.45",
                        "country": "Iran",
                        "port": 443,
                        "timestamp": "2025-12-01T10:30:00Z",
                    },
                    {
                        "src_ip": "185.220.101.53",
                        "dst_ip": "203.0.113.46",
                        "country": "Iran",
                        "port": 80,
                        "timestamp": "2025-12-02T14:15:00Z",
                    },
                ],
                "countries": ["Iran"],
                "ports": ["443", "80"],
            }
        }

        # Extract entities as supervisor would
        entities = _extract_entities_from_previous_results(aggregated)

        # Verify entities found (order independent for IPs)
        assert set(entities["ips"]) == {"185.220.101.52", "185.220.101.53"}
        assert "Iran" in entities["countries"]
        assert "443" in entities["ports"]

        # Build enriched question for threat_analyst
        original_question = "what is the reputation of those IPs?"
        enriched_question = _build_context_aware_threat_question(original_question, entities)

        # Verify enrichment
        assert "185.220.101.52" in enriched_question
        assert "185.220.101.53" in enriched_question
        assert "Iran" in enriched_question
        assert "443" in enriched_question
        assert "Previously discovered entities" in enriched_question

        # Now threat_analyst receives this enriched question with actual IPs,
        # so it can look up reputation for THOSE SPECIFIC IPs instead of
        # misinterpreting the question and doing a generic lookup.
        print("\nEnriched question passed to threat_analyst:")
        print(enriched_question)

        # This enriched context ensures threat_analyst focuses on the
        # actual discovered IPs rather than getting confused.
        assert len(enriched_question) > len(original_question)
