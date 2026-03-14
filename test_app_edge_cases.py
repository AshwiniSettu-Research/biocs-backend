"""
Edge case tests for the alignment API and UniProt endpoint.
Extends coverage from test_app.py without modifying it.

Run with: python3 -m pytest test_app_edge_cases.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import ConnectionError as RequestsConnectionError
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestIdenticalSequences:
    """Test alignment of identical sequences."""

    def test_identical_sequences(self, client):
        """Aligning a sequence with itself should yield 100% identity."""
        seq = "MVLSPADKTN"
        resp = client.post("/api/align", json={
            "seq1": seq,
            "seq2": seq,
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        result = data["results"][0]

        assert result["identity"] == 100.0
        assert result["mismatches"] == 0
        assert result["gaps"] == 0

        # All position scores should be identical
        for pos in result["position_scores"]:
            assert pos["category"] == "identical"
            assert pos["conservation"] == 1.0

        # Scoring breakdown should show 100% identical
        bd = result["scoring_breakdown"]
        assert bd["identical"] == result["alignment_length"]
        assert bd["identical_pct"] == 100.0


class TestCompletelyDifferentSequences:
    """Test alignment of completely different sequences."""

    def test_completely_different_sequences(self, client):
        """Aligning all-A with all-W should yield 0% identity."""
        resp = client.post("/api/align", json={
            "seq1": "AAAAAAAAAA",
            "seq2": "WWWWWWWWWW",
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        result = data["results"][0]

        # Identity should be 0 since no positions match
        assert result["identity"] == 0.0

        # No position should be identical
        for pos in result["position_scores"]:
            if pos["res1"] != "-" and pos["res2"] != "-":
                assert pos["category"] != "identical"


class TestVeryLongSequences:
    """Test alignment with long sequences."""

    def test_very_long_sequences(self, client):
        """Aligning two 500-char sequences should complete and return valid structure."""
        seq1 = ("MVLSPADKTN" * 50)  # 500 chars
        seq2 = ("MVHLTPEEKS" * 50)  # 500 chars

        resp = client.post("/api/align", json={
            "seq1": seq1,
            "seq2": seq2,
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        result = data["results"][0]

        assert "aligned_seq1" in result
        assert "aligned_seq2" in result
        assert "score" in result
        assert "identity" in result
        assert result["seq1_length"] == 500
        assert result["seq2_length"] == 500
        assert result["alignment_length"] > 0


class TestPositionScoresTruncation:
    """Test position_scores truncation for long alignments."""

    def test_position_scores_truncation(self, client):
        """Sequences > 500 chars should trigger position_scores truncation."""
        # Build sequences slightly over 500 residues
        seq1 = "MVLSPADKTN" * 51  # 510 chars
        seq2 = "MVHLTPEEKS" * 51  # 510 chars

        resp = client.post("/api/align", json={
            "seq1": seq1,
            "seq2": seq2,
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        result = data["results"][0]

        # The alignment length will be >= 510, so truncation should apply
        assert result["position_scores_truncated"] is True
        assert len(result["position_scores"]) <= 500


class TestConservationPlotShortSequences:
    """Test conservation plot with very short sequences."""

    def test_conservation_plot_short_sequences(self, client):
        """Aligning 'MV' with 'MV' should handle window_size edge case."""
        resp = client.post("/api/align", json={
            "seq1": "MV",
            "seq2": "MV",
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        result = data["results"][0]

        # conservation_plot should still be a list (may be empty or have 1 point)
        assert isinstance(result["conservation_plot"], list)
        # With two identical residues and window_size=min(10, 2)=2, we get 1 data point
        if len(result["conservation_plot"]) > 0:
            for pt in result["conservation_plot"]:
                assert "position" in pt
                assert "identity" in pt


class TestUniProtEndpoint:
    """Tests for the UniProt fetch endpoint with mocking."""

    @patch("app.http_requests.get")
    def test_uniprot_endpoint_success(self, mock_get, client):
        """Mock a valid UniProt JSON response and verify parsed fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "uniProtkbId": "HBA_HUMAN",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Hemoglobin subunit alpha"},
                },
            },
            "organism": {
                "scientificName": "Homo sapiens",
            },
            "genes": [
                {"geneName": {"value": "HBA1"}},
            ],
            "sequence": {
                "value": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
                "length": 50,
            },
            "comments": [
                {
                    "commentType": "FUNCTION",
                    "texts": [{"value": "Involved in oxygen transport."}],
                },
            ],
            "uniProtKBCrossReferences": [
                {"database": "PDB", "id": "1A3N"},
                {"database": "GO", "id": "GO:0005833"},
            ],
        }
        mock_get.return_value = mock_response

        resp = client.get("/api/uniprot/P69905")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["accession"] == "P69905"
        assert data["entry_name"] == "HBA_HUMAN"
        assert data["protein_name"] == "Hemoglobin subunit alpha"
        assert data["organism"] == "Homo sapiens"
        assert data["gene_name"] == "HBA1"
        assert "MVLSPADKTN" in data["sequence"]
        assert data["length"] == 50
        assert data["function"] == "Involved in oxygen transport."
        assert "PDB" in data["cross_references"]
        assert "1A3N" in data["cross_references"]["PDB"]

    @patch("app.http_requests.get")
    def test_uniprot_endpoint_not_found(self, mock_get, client):
        """Mock a 404 from UniProt and verify error response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        resp = client.get("/api/uniprot/INVALID999")

        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    @patch("app.http_requests.get")
    def test_uniprot_endpoint_server_error(self, mock_get, client):
        """Mock a ConnectionError and verify generic error response (no details)."""
        mock_get.side_effect = RequestsConnectionError("Network unreachable: internal detail")

        resp = client.get("/api/uniprot/P69905")

        assert resp.status_code == 502
        data = resp.get_json()
        assert "error" in data
        # Should not leak internal exception details
        assert "Network unreachable" not in data["error"]
        assert "internal detail" not in data["error"]


class TestAlignEndpointNoJsonContentType:
    """Test POST /api/align without proper JSON content type."""

    def test_align_endpoint_no_json_content_type(self, client):
        """POST without JSON content type should return 400."""
        resp = client.post(
            "/api/align",
            data="seq1=MVLSPADKTN&seq2=MVHLTPEEKS",
            content_type="application/x-www-form-urlencoded",
        )

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


class TestScoringBreakdownFields:
    """Test that scoring breakdown has all expected fields."""

    def test_scoring_breakdown_has_all_fields(self, client):
        """Verify breakdown has all count and percentage fields."""
        resp = client.post("/api/align", json={
            "seq1": "MVLSPADKTN",
            "seq2": "MVHLTPEEKS",
            "params": {},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        bd = data["results"][0]["scoring_breakdown"]

        # Count fields
        assert "identical" in bd
        assert "conservative" in bd
        assert "semi_conservative" in bd
        assert "non_conservative" in bd
        assert "gaps" in bd
        assert "total" in bd

        # Percentage fields
        assert "identical_pct" in bd
        assert "conservative_pct" in bd
        assert "semi_conservative_pct" in bd
        assert "non_conservative_pct" in bd
        assert "gap_pct" in bd
        assert "similarity_pct" in bd

        # All counts should be non-negative integers
        for key in ["identical", "conservative", "semi_conservative", "non_conservative", "gaps", "total"]:
            assert isinstance(bd[key], int)
            assert bd[key] >= 0

        # All percentages should be non-negative floats
        for key in ["identical_pct", "conservative_pct", "semi_conservative_pct",
                     "non_conservative_pct", "gap_pct", "similarity_pct"]:
            assert isinstance(bd[key], float)
            assert bd[key] >= 0.0

        # Percentages (excluding similarity) should sum to ~100%
        pct_sum = (
            bd["identical_pct"]
            + bd["conservative_pct"]
            + bd["semi_conservative_pct"]
            + bd["non_conservative_pct"]
            + bd["gap_pct"]
        )
        assert abs(pct_sum - 100.0) < 0.1

        # Similarity should be >= identical (it includes conservative)
        assert bd["similarity_pct"] >= bd["identical_pct"]

        # Counts should sum to total
        count_sum = bd["identical"] + bd["conservative"] + bd["semi_conservative"] + bd["non_conservative"] + bd["gaps"]
        assert count_sum == bd["total"]
