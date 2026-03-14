"""
BIOCS Backend API
Flask server for protein sequence alignment using CM-BLOSUM-NW algorithm.
"""

import os
import re
import math
import logging
import requests as http_requests
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import get_config
from cm_blosum_nw import CM_BLOSUM_NW, AMINO_ACIDS as CM_AMINO_ACIDS, BLOSUM62_DATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app_config = get_config()
app.config.from_object(app_config)
CORS(app, resources={r"/api/*": {
    "origins": app_config.CORS_ORIGINS,
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})

# Defer mlpt import to avoid slow torch import at startup
from mlpt.routes import mlpt_bp  # noqa: E402
app.register_blueprint(mlpt_bp)

VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(seq, name="Sequence"):
    """Validate a protein sequence string against the 20 standard amino acids."""
    if not seq or not seq.strip():
        return None, f"{name} is empty."
    cleaned = re.sub(r'[^A-Za-z]', '', seq.upper())
    if len(cleaned) == 0:
        return None, f"{name} contains no valid characters."
    if len(cleaned) > 10000:
        return None, f"{name} exceeds maximum length of 10,000 residues."
    if len(cleaned) < 2:
        return None, f"{name} must be at least 2 residues long."
    invalid_chars = set(cleaned) - VALID_PROTEIN_CHARS
    if invalid_chars:
        sorted_invalid = sorted(invalid_chars)
        return None, f"{name} contains invalid amino acid characters: {', '.join(sorted_invalid)}. Only the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY) are accepted."
    return cleaned, None


def parse_fasta_text(text):
    """Parse FASTA-formatted text and extract the first sequence."""
    lines = text.strip().split('\n')
    seq_parts = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq_parts:
                break
            continue
        seq_parts.append(line)
    return ''.join(seq_parts)


def compute_amino_acid_composition(seq):
    """Compute amino acid composition (frequency) for a sequence."""
    counts = Counter(seq)
    length = len(seq)
    composition = {}
    for aa in CM_AMINO_ACIDS:
        composition[aa] = round(counts.get(aa, 0) / length, 4) if length > 0 else 0.0
    return composition


def compute_dipeptide_composition(seq):
    """Compute dipeptide composition for a sequence."""
    if len(seq) < 2:
        return {}
    total_dipeptides = len(seq) - 1
    counts = Counter()
    for i in range(len(seq) - 1):
        dp = seq[i] + seq[i + 1]
        counts[dp] += 1
    composition = {}
    for dp, count in counts.most_common(20):
        composition[dp] = round(count / total_dipeptides, 4)
    return composition


def compute_information_content(seq):
    """Compute IC values per amino acid in the sequence."""
    epsilon = 1e-6
    counts = Counter(seq)
    length = len(seq)
    ic_values = {}
    for aa in CM_AMINO_ACIDS:
        freq = counts.get(aa, 0) / length if length > 0 else 0.0
        ic = -math.log2(freq + epsilon) if freq > 0 else -math.log2(epsilon)
        ic_values[aa] = round(ic, 4)
    return ic_values


def compute_per_position_scores(aligned1, aligned2):
    """Compute per-position BLOSUM62, category, and conservation scores.

    Each position gets a 'category' field: one of 'identical', 'conservative',
    'semi_conservative', 'non_conservative', or 'gap'. This is the single source
    of truth for classification — compute_scoring_breakdown counts from it.
    """
    positions = []
    for i in range(len(aligned1)):
        c1 = aligned1[i]
        c2 = aligned2[i]
        pos_data = {"pos": i + 1, "res1": c1, "res2": c2}

        if c1 == '-' or c2 == '-':
            pos_data["category"] = "gap"
            pos_data["blosum62"] = None
            pos_data["conservation"] = 0
        elif c1 == c2:
            pos_data["category"] = "identical"
            pos_data["blosum62"] = BLOSUM62_DATA.get(c1, {}).get(c2, 0)
            pos_data["conservation"] = 1.0
        else:
            score = BLOSUM62_DATA.get(c1, {}).get(c2, 0)
            pos_data["blosum62"] = score
            if score >= 1:
                pos_data["category"] = "conservative"
                pos_data["conservation"] = 0.6
            elif score >= 0:
                pos_data["category"] = "semi_conservative"
                pos_data["conservation"] = 0.3
            else:
                pos_data["category"] = "non_conservative"
                pos_data["conservation"] = 0.0

        positions.append(pos_data)
    return positions


def compute_scoring_breakdown(position_scores):
    """Compute scoring category breakdown by counting from position_scores.

    Consumes the 'category' field set by compute_per_position_scores,
    rather than re-deriving classification from raw scores.
    """
    counts = Counter(p["category"] for p in position_scores)
    total = len(position_scores)
    identical = counts.get("identical", 0)
    conservative = counts.get("conservative", 0)
    semi_conservative = counts.get("semi_conservative", 0)
    non_conservative = counts.get("non_conservative", 0)
    gap_count = counts.get("gap", 0)

    return {
        "identical": identical,
        "conservative": conservative,
        "semi_conservative": semi_conservative,
        "non_conservative": non_conservative,
        "gaps": gap_count,
        "total": total,
        "identical_pct": round(100.0 * identical / total, 2) if total else 0,
        "conservative_pct": round(100.0 * conservative / total, 2) if total else 0,
        "semi_conservative_pct": round(100.0 * semi_conservative / total, 2) if total else 0,
        "non_conservative_pct": round(100.0 * non_conservative / total, 2) if total else 0,
        "gap_pct": round(100.0 * gap_count / total, 2) if total else 0,
        "similarity_pct": round(100.0 * (identical + conservative) / total, 2) if total else 0,
    }


def compute_sliding_window_identity(aligned1, aligned2, window_size=10):
    """Compute sliding window identity for conservation plot. O(n) sliding window."""
    n = len(aligned1)
    if n < window_size:
        return []

    # Initialize the first window
    matches = 0
    valid = 0
    for j in range(window_size):
        c1, c2 = aligned1[j], aligned2[j]
        if c1 != '-' and c2 != '-':
            valid += 1
            if c1 == c2:
                matches += 1

    points = []
    identity = round(100.0 * matches / valid, 1) if valid > 0 else 0
    points.append({"position": 1, "identity": identity})

    # Slide the window: remove leftmost element, add new rightmost element
    for i in range(1, n - window_size + 1):
        # Remove element leaving the window (position i-1)
        old_c1, old_c2 = aligned1[i - 1], aligned2[i - 1]
        if old_c1 != '-' and old_c2 != '-':
            valid -= 1
            if old_c1 == old_c2:
                matches -= 1

        # Add element entering the window (position i + window_size - 1)
        new_c1, new_c2 = aligned1[i + window_size - 1], aligned2[i + window_size - 1]
        if new_c1 != '-' and new_c2 != '-':
            valid += 1
            if new_c1 == new_c2:
                matches += 1

        identity = round(100.0 * matches / valid, 1) if valid > 0 else 0
        points.append({"position": i + 1, "identity": identity})

    return points


def validate_params(params):
    """Validate and parse alignment parameters. Returns (parsed_params, error)."""
    try:
        alpha = float(params.get('alpha', 0.5))
        beta = float(params.get('beta', 0.3))
        gap_open = float(params.get('gap_open', -10.0))
        gap_extend = float(params.get('gap_extend', -1.0))
        bandwidth = int(params.get('bandwidth', 5))
    except (ValueError, TypeError) as e:
        return None, f"Invalid parameter value: {e}"

    errors = []
    if not (0 <= alpha <= 1):
        errors.append(f"Alpha must be between 0 and 1 (got {alpha}).")
    if not (0 <= beta <= 1):
        errors.append(f"Beta must be between 0 and 1 (got {beta}).")
    if not math.isfinite(gap_open):
        errors.append(f"Gap open must be a finite number (got {gap_open}).")
    if not math.isfinite(gap_extend):
        errors.append(f"Gap extend must be a finite number (got {gap_extend}).")
    if not (1 <= bandwidth <= 50):
        errors.append(f"Bandwidth must be between 1 and 50 (got {bandwidth}).")

    if errors:
        return None, " ".join(errors)

    return {
        "alpha": alpha,
        "beta": beta,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "bandwidth": bandwidth,
    }, None


def run_cm_blosum_nw(seq1, seq2, params):
    """Run CM-BLOSUM-NW alignment and return enriched results."""
    alpha = params['alpha']
    beta = params['beta']
    gap_open = params['gap_open']
    gap_extend = params['gap_extend']
    bandwidth = params['bandwidth']

    aligner = CM_BLOSUM_NW(
        alpha=alpha,
        beta=beta,
        gap_open=gap_open,
        gap_extend=gap_extend,
        bandwidth=bandwidth,
    )
    result = aligner.align_pair(seq1, seq2, verbose=False)

    aligned1 = result["aligned_seq1"]
    aligned2 = result["aligned_seq2"]

    # Compositional analysis
    aac_seq1 = compute_amino_acid_composition(seq1)
    aac_seq2 = compute_amino_acid_composition(seq2)
    dpc_seq1 = compute_dipeptide_composition(seq1)
    dpc_seq2 = compute_dipeptide_composition(seq2)
    ic_seq1 = compute_information_content(seq1)
    ic_seq2 = compute_information_content(seq2)

    # Per-position analysis (position_scores is single source of truth for classification)
    position_scores = compute_per_position_scores(aligned1, aligned2)
    scoring_breakdown = compute_scoring_breakdown(position_scores)
    conservation_plot = compute_sliding_window_identity(aligned1, aligned2, window_size=min(10, len(aligned1)))

    # Truncate position_scores for large alignments to reduce payload size
    MAX_POSITION_SCORES = 500
    position_scores_truncated = len(position_scores) > MAX_POSITION_SCORES
    position_scores_sent = position_scores[:MAX_POSITION_SCORES] if position_scores_truncated else position_scores

    return {
        "algorithm": "CM-BLOSUM-NW",
        "aligned_seq1": aligned1,
        "aligned_seq2": aligned2,
        "score": round(float(result["score"]), 4) if result["score"] is not None else None,
        "identity": round(result["identity"], 2),
        "matches": int(result["matches"]),
        "mismatches": int(result["mismatches"]),
        "gaps": int(result["gaps"]),
        "gap_opens": int(result.get("gap_opens", 0)),
        "alignment_length": int(result["alignment_length"]),
        "runtime_seconds": round(result["runtime_seconds"], 6),
        "memory_peak_mb": round(result["memory_peak_mb"], 4) if result["memory_peak_mb"] is not None else None,
        "seq1_length": len(seq1),
        "seq2_length": len(seq2),
        "bandwidth_used": int(result.get("bandwidth_used", bandwidth)),
        "params_used": {
            "alpha": alpha,
            "beta": beta,
            "gap_open": gap_open,
            "gap_extend": gap_extend,
            "bandwidth": bandwidth,
        },
        # Advanced analysis data
        "compositional_analysis": {
            "seq1_aac": aac_seq1,
            "seq2_aac": aac_seq2,
            "seq1_dpc_top": dpc_seq1,
            "seq2_dpc_top": dpc_seq2,
            "seq1_ic": ic_seq1,
            "seq2_ic": ic_seq2,
        },
        "scoring_breakdown": scoring_breakdown,
        "position_scores": position_scores_sent,
        "position_scores_truncated": position_scores_truncated,
        "conservation_plot": conservation_plot,
    }


@app.route('/api/align', methods=['POST'])
def align():
    """Main alignment endpoint."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        raw_seq1 = data.get('seq1', '')
        raw_seq2 = data.get('seq2', '')
        raw_params = data.get('params', {})

        # Handle FASTA input
        if raw_seq1.strip().startswith('>'):
            raw_seq1 = parse_fasta_text(raw_seq1)
        if raw_seq2.strip().startswith('>'):
            raw_seq2 = parse_fasta_text(raw_seq2)

        # Validate sequences
        seq1, err1 = validate_sequence(raw_seq1, "Sequence 1")
        if err1:
            return jsonify({"error": err1}), 400

        seq2, err2 = validate_sequence(raw_seq2, "Sequence 2")
        if err2:
            return jsonify({"error": err2}), 400

        # Validate parameters
        params, param_err = validate_params(raw_params)
        if param_err:
            return jsonify({"error": param_err}), 400

        logger.info(f"Running CM-BLOSUM-NW: seq1={len(seq1)}aa, seq2={len(seq2)}aa")
        cm_result = run_cm_blosum_nw(seq1, seq2, params)

        return jsonify({"results": [cm_result]})

    except Exception as e:
        logger.error(f"Alignment error: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred during alignment. Please try again."}), 500


@app.route('/api/uniprot/<accession>', methods=['GET'])
def fetch_uniprot(accession):
    """Fetch protein sequence and metadata from UniProt."""
    try:
        # Validate accession format
        accession = accession.strip().upper()
        if not re.match(r'^[A-Z0-9_]+$', accession):
            return jsonify({"error": "Invalid UniProt accession format."}), 400

        # Fetch from UniProt REST API
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        resp = http_requests.get(url, timeout=10)

        if resp.status_code == 404:
            return jsonify({"error": f"UniProt entry '{accession}' not found."}), 404
        if resp.status_code != 200:
            return jsonify({"error": f"UniProt API error (HTTP {resp.status_code})."}), 502

        data = resp.json()

        # Extract sequence
        sequence = data.get("sequence", {}).get("value", "")
        if not sequence:
            return jsonify({"error": "No sequence found in UniProt entry."}), 404

        # Extract metadata
        entry_name = data.get("uniProtkbId", accession)
        protein_name = ""
        protein_desc = data.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        if rec_name:
            full_name = rec_name.get("fullName", {})
            protein_name = full_name.get("value", "") if isinstance(full_name, dict) else str(full_name)
        elif protein_desc.get("submissionNames"):
            sub_name = protein_desc["submissionNames"][0].get("fullName", {})
            protein_name = sub_name.get("value", "") if isinstance(sub_name, dict) else str(sub_name)

        organism = ""
        org_data = data.get("organism", {})
        if org_data.get("scientificName"):
            organism = org_data["scientificName"]

        gene_name = ""
        genes = data.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {}).get("value", "")

        seq_length = data.get("sequence", {}).get("length", len(sequence))

        # Extract functional info
        function_text = ""
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                break

        # Database cross-references
        cross_refs = {}
        db_refs = data.get("uniProtKBCrossReferences", [])
        for ref in db_refs:
            db = ref.get("database", "")
            ref_id = ref.get("id", "")
            if db in ("PDB", "Pfam", "InterPro", "GO"):
                if db not in cross_refs:
                    cross_refs[db] = []
                if len(cross_refs[db]) < 5:
                    cross_refs[db].append(ref_id)

        return jsonify({
            "accession": accession,
            "entry_name": entry_name,
            "protein_name": protein_name,
            "organism": organism,
            "gene_name": gene_name,
            "sequence": sequence,
            "length": seq_length,
            "function": function_text,
            "cross_references": cross_refs,
        })

    except http_requests.exceptions.Timeout:
        return jsonify({"error": "UniProt request timed out."}), 504
    except http_requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to UniProt."}), 502
    except Exception as e:
        logger.error(f"UniProt fetch error: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred fetching UniProt data. Please try again."}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "algorithm": "CM-BLOSUM-NW"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=app.config.get('DEBUG', False))
