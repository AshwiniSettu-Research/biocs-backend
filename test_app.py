"""
Tests for BIOCS Backend API.
Run with: python3 -m pytest test_app.py -v
"""

import json
import pytest
from app import (
    app,
    validate_sequence,
    validate_params,
    parse_fasta_text,
    compute_amino_acid_composition,
    compute_dipeptide_composition,
    compute_information_content,
    compute_per_position_scores,
    compute_scoring_breakdown,
    compute_sliding_window_identity,
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# === validate_sequence ===

class TestValidateSequence:
    def test_valid_sequence(self):
        seq, err = validate_sequence("MVLSPADKTN", "Seq")
        assert err is None
        assert seq == "MVLSPADKTN"

    def test_uppercase_conversion(self):
        seq, err = validate_sequence("mvlspadktn", "Seq")
        assert err is None
        assert seq == "MVLSPADKTN"

    def test_strips_non_alpha(self):
        seq, err = validate_sequence("MVL SPA 123 DKT\nN", "Seq")
        assert err is None
        assert seq == "MVLSPADKTN"

    def test_empty_string(self):
        seq, err = validate_sequence("", "Seq")
        assert seq is None
        assert "empty" in err

    def test_whitespace_only(self):
        seq, err = validate_sequence("   \n\t  ", "Seq")
        assert seq is None
        assert "empty" in err

    def test_none_input(self):
        seq, err = validate_sequence(None, "Seq")
        assert seq is None
        assert "empty" in err

    def test_no_valid_characters(self):
        seq, err = validate_sequence("12345!@#$%", "Seq")
        assert seq is None
        assert "no valid characters" in err

    def test_too_short(self):
        seq, err = validate_sequence("M", "Seq")
        assert seq is None
        assert "at least 2" in err

    def test_exactly_two_residues(self):
        seq, err = validate_sequence("MV", "Seq")
        assert err is None
        assert seq == "MV"

    def test_too_long(self):
        seq, err = validate_sequence("A" * 10001, "Seq")
        assert seq is None
        assert "10,000" in err

    def test_exactly_max_length(self):
        seq, err = validate_sequence("A" * 10000, "Seq")
        assert err is None
        assert len(seq) == 10000

    def test_invalid_amino_acids(self):
        seq, err = validate_sequence("MVLSXZUO", "Seq")
        assert seq is None
        assert "invalid amino acid" in err
        assert "O" in err
        assert "U" in err
        assert "X" in err
        assert "Z" in err

    def test_all_valid_amino_acids(self):
        seq, err = validate_sequence("ACDEFGHIKLMNPQRSTVWY", "Seq")
        assert err is None
        assert seq == "ACDEFGHIKLMNPQRSTVWY"

    def test_error_includes_name(self):
        _, err = validate_sequence("", "Sequence 1")
        assert "Sequence 1" in err


# === parse_fasta_text ===

class TestParseFastaText:
    def test_simple_fasta(self):
        text = ">header\nMVLSPADK\nTNVKAAWG"
        assert parse_fasta_text(text) == "MVLSPADKTNVKAAWG"

    def test_multiple_sequences_returns_first(self):
        text = ">seq1\nMVLS\n>seq2\nACDE"
        assert parse_fasta_text(text) == "MVLS"

    def test_no_header(self):
        text = "MVLSPADKTN"
        assert parse_fasta_text(text) == "MVLSPADKTN"

    def test_empty_sequence(self):
        text = ">header only"
        assert parse_fasta_text(text) == ""

    def test_whitespace_handling(self):
        text = ">header\n  MVLS  \n  PADK  "
        assert parse_fasta_text(text) == "MVLSPADK"


# === compute_amino_acid_composition ===

class TestAminoAcidComposition:
    def test_simple_sequence(self):
        comp = compute_amino_acid_composition("AACC")
        assert comp["A"] == 0.5
        assert comp["C"] == 0.5
        assert comp["D"] == 0.0

    def test_all_same(self):
        comp = compute_amino_acid_composition("AAAA")
        assert comp["A"] == 1.0
        for aa in "CDEFGHIKLMNPQRSTVWY":
            assert comp[aa] == 0.0


# === compute_dipeptide_composition ===

class TestDipeptideComposition:
    def test_simple(self):
        comp = compute_dipeptide_composition("AACCA")
        assert "AA" in comp or "AC" in comp
        total = sum(comp.values())
        assert abs(total - 1.0) < 0.01

    def test_too_short(self):
        comp = compute_dipeptide_composition("A")
        assert comp == {}

    def test_top_20_limit(self):
        # 20 amino acids * 20 = 400 possible dipeptides, but we only return top 20
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 10
        comp = compute_dipeptide_composition(long_seq)
        assert len(comp) <= 20


# === compute_information_content ===

class TestInformationContent:
    def test_returns_all_amino_acids(self):
        ic = compute_information_content("MVLSPADKTN")
        assert len(ic) == 20  # All 20 standard AAs

    def test_present_aa_has_lower_ic_than_absent(self):
        ic = compute_information_content("AAAAAA")
        # A is very frequent, should have low IC
        # C is absent, should have very high IC (near -log2(epsilon))
        assert ic["A"] < ic["C"]

    def test_all_values_positive(self):
        ic = compute_information_content("MVLSPADKTN")
        for val in ic.values():
            assert val > 0


# === compute_per_position_scores ===

class TestPerPositionScores:
    def test_identical_positions(self):
        positions = compute_per_position_scores("MVLS", "MVLS")
        for p in positions:
            assert p["category"] == "identical"
            assert p["conservation"] == 1.0

    def test_gap_positions(self):
        positions = compute_per_position_scores("M-LS", "MVLS")
        assert positions[1]["category"] == "gap"
        assert positions[1]["blosum62"] is None

    def test_mismatch_positions(self):
        positions = compute_per_position_scores("MVLS", "AVLT")
        # Position 0: M vs A = mismatch
        assert positions[0]["category"] in ("conservative", "semi_conservative", "non_conservative")
        assert positions[0]["res1"] == "M"
        assert positions[0]["res2"] == "A"

    def test_position_numbering_starts_at_1(self):
        positions = compute_per_position_scores("MV", "MV")
        assert positions[0]["pos"] == 1
        assert positions[1]["pos"] == 2

    def test_all_categories_present(self):
        # Verify category field is always set
        positions = compute_per_position_scores("MVLS-", "AVLT-")
        for p in positions:
            assert "category" in p
            assert p["category"] in ("identical", "conservative", "semi_conservative", "non_conservative", "gap")


# === compute_scoring_breakdown ===

class TestScoringBreakdown:
    def test_all_identical(self):
        positions = compute_per_position_scores("MVLS", "MVLS")
        bd = compute_scoring_breakdown(positions)
        assert bd["identical"] == 4
        assert bd["total"] == 4
        assert bd["identical_pct"] == 100.0
        assert bd["gaps"] == 0

    def test_all_gaps(self):
        positions = compute_per_position_scores("----", "MVLS")
        bd = compute_scoring_breakdown(positions)
        assert bd["gaps"] == 4
        assert bd["identical"] == 0
        assert bd["gap_pct"] == 100.0

    def test_percentages_sum_to_100(self):
        positions = compute_per_position_scores("MVL-S", "AVL-T")
        bd = compute_scoring_breakdown(positions)
        total_pct = (bd["identical_pct"] + bd["conservative_pct"] +
                     bd["semi_conservative_pct"] + bd["non_conservative_pct"] +
                     bd["gap_pct"])
        assert abs(total_pct - 100.0) < 0.1

    def test_similarity_includes_conservative(self):
        positions = compute_per_position_scores("MVLS", "MVLS")
        bd = compute_scoring_breakdown(positions)
        assert bd["similarity_pct"] >= bd["identical_pct"]


# === compute_sliding_window_identity ===

class TestSlidingWindowIdentity:
    def test_all_identical(self):
        points = compute_sliding_window_identity("MVLSPADK", "MVLSPADK", window_size=4)
        for p in points:
            assert p["identity"] == 100.0

    def test_no_identity(self):
        points = compute_sliding_window_identity("AAAA", "CCCC", window_size=2)
        for p in points:
            assert p["identity"] == 0.0

    def test_position_numbering(self):
        points = compute_sliding_window_identity("MVLS", "MVLS", window_size=2)
        assert points[0]["position"] == 1
        assert len(points) == 3  # len - window_size + 1


# === validate_params ===

class TestValidateParams:
    def test_defaults(self):
        params, err = validate_params({})
        assert err is None
        assert params["alpha"] == 0.5
        assert params["beta"] == 0.3
        assert params["bandwidth"] == 5

    def test_valid_custom_params(self):
        params, err = validate_params({
            "alpha": 0.8, "beta": 0.1, "gap_open": -12.0,
            "gap_extend": -2.0, "bandwidth": 10,
        })
        assert err is None
        assert params["alpha"] == 0.8

    def test_alpha_out_of_range(self):
        _, err = validate_params({"alpha": 1.5})
        assert err is not None
        assert "Alpha" in err

    def test_beta_out_of_range(self):
        _, err = validate_params({"beta": -0.1})
        assert err is not None
        assert "Beta" in err

    def test_bandwidth_zero(self):
        _, err = validate_params({"bandwidth": 0})
        assert err is not None
        assert "Bandwidth" in err

    def test_bandwidth_too_large(self):
        _, err = validate_params({"bandwidth": 100})
        assert err is not None
        assert "Bandwidth" in err

    def test_zero_alpha_is_valid(self):
        params, err = validate_params({"alpha": 0})
        assert err is None
        assert params["alpha"] == 0

    def test_non_numeric_value(self):
        _, err = validate_params({"alpha": "abc"})
        assert err is not None
        assert "Invalid parameter" in err


# === API Endpoints ===

class TestAlignEndpoint:
    def test_successful_alignment(self, client):
        resp = client.post("/api/align", json={
            "seq1": "MVLSPADKTN",
            "seq2": "MVHLTPEEKS",
            "params": {},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["algorithm"] == "CM-BLOSUM-NW"
        assert "aligned_seq1" in result
        assert "aligned_seq2" in result
        assert "score" in result
        assert "identity" in result
        assert "compositional_analysis" in result
        assert "scoring_breakdown" in result
        assert "position_scores" in result
        assert "conservation_plot" in result

    def test_fasta_input(self, client):
        resp = client.post("/api/align", json={
            "seq1": ">seq1\nMVLSPADKTN",
            "seq2": ">seq2\nMVHLTPEEKS",
            "params": {},
        })
        assert resp.status_code == 200

    def test_missing_sequences(self, client):
        resp = client.post("/api/align", json={
            "seq1": "",
            "seq2": "MVLSPADKTN",
            "params": {},
        })
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_invalid_amino_acids(self, client):
        resp = client.post("/api/align", json={
            "seq1": "MVLSXZPADKTN",
            "seq2": "MVHLTPEEKS",
            "params": {},
        })
        assert resp.status_code == 400
        assert "invalid amino acid" in resp.get_json()["error"]

    def test_no_json_body(self, client):
        resp = client.post("/api/align", content_type="application/json", data="")
        assert resp.status_code == 400

    def test_invalid_params_rejected(self, client):
        resp = client.post("/api/align", json={
            "seq1": "MVLSPADKTN",
            "seq2": "MVHLTPEEKS",
            "params": {"alpha": 5.0},
        })
        assert resp.status_code == 400
        assert "Alpha" in resp.get_json()["error"]

    def test_position_scores_have_category(self, client):
        resp = client.post("/api/align", json={
            "seq1": "MVLSPADKTN",
            "seq2": "MVHLTPEEKS",
            "params": {},
        })
        data = resp.get_json()
        for pos in data["results"][0]["position_scores"]:
            assert "category" in pos
            assert pos["category"] in ("identical", "conservative", "semi_conservative", "non_conservative", "gap")


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["algorithm"] == "CM-BLOSUM-NW"
