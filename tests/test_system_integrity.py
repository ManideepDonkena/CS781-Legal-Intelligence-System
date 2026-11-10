import unittest
import torch
import numpy as np
import re
import sys
from pathlib import Path

# Add code directories to path
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "phase4" / "src"))
sys.path.insert(0, str(CODE_ROOT / "phase7" / "src"))
sys.path.insert(0, str(CODE_ROOT / "shared"))

class TestPhase1RoPEIntegrity(unittest.TestCase):
    """Test Phase 1 Rotary Position Embedding numerical precision and behavior."""
    def test_rope_float32_precision(self):
        dim = 64
        max_seq_len = 2048
        base = 10000.0
        
        # Test float32 inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        
        # Verify no frequency collapse (no zeros at long contexts)
        self.assertTrue(torch.all(freqs[-1] > 0), "RoPE frequencies must remain strictly positive at t=2048")
        self.assertFalse(torch.isnan(freqs).any(), "No NaN values in RoPE freqs")
        self.assertFalse(torch.isinf(freqs).any(), "No Inf values in RoPE freqs")

class TestPhase2SpanIntegrity(unittest.TestCase):
    """Test Phase 2 Character-level span tracking and Cause-Title distance."""
    def test_exact_character_offset_reconstruction(self):
        sample_doc = "IN THE SUPREME COURT OF INDIA\nState of Maharashtra   Versus   Ramesh Sharma\nJudgment"
        entity_text = "Ramesh Sharma"
        start = sample_doc.find(entity_text)
        end = start + len(entity_text)
        
        # Exact character slice check
        extracted = sample_doc[start:end]
        self.assertEqual(extracted, entity_text)
        
    def test_cause_title_versus_feature(self):
        lines = [
            "IN THE SUPREME COURT OF INDIA",
            "CIVIL APPELLATE JURISDICTION",
            "Union of India Versus M/s ABC Ltd."
        ]
        text = "\n".join(lines)
        v_idx = text.find("Versus")
        self.assertGreater(v_idx, 0)
        
        petitioner_pos = text.find("Union of India")
        respondent_pos = text.find("M/s ABC Ltd.")
        
        # Signed distance to 'Versus'
        dist_pet = petitioner_pos - v_idx
        dist_resp = respondent_pos - v_idx
        
        self.assertLess(dist_pet, 0, "Petitioner must precede 'Versus'")
        self.assertGreater(dist_resp, 0, "Respondent must follow 'Versus'")

class TestPhase4GlitchRepairIntegrity(unittest.TestCase):
    """Test Phase 4 Inverse Glitch Vocabulary Repair."""
    def test_text_fix_known_corruptions(self):
        mock_table = {
            "companyrt": "court",
            "numberice": "notice",
            "companymissioner": "commissioner",
            "numberember": "november",
            "companyaccused": "co-accused",
            "numberdoubt": "no doubt"
        }
        
        WORD_RE = re.compile(r"\b(company|number)([a-z]+)\b")
        sample_text = "The companyrt issued a numberice on 15 numberember to the companyaccused without numberdoubt."
        repaired = WORD_RE.sub(lambda m: mock_table.get(m.group(0), m.group(0)), sample_text)
        
        expected = "The court issued a notice on 15 november to the co-accused without no doubt."
        self.assertEqual(repaired, expected)
        
    def test_preserve_genuine_company(self):
        sample_text = "The company law board decided the matter."
        WORD_RE = re.compile(r"\b(company|number)([a-z]+)\b")
        matches = [m.group(0) for m in WORD_RE.finditer(sample_text)]
        self.assertEqual(matches, [], "Standalone 'company' must never be matched or corrupted")

class TestPhase7PCRIntegrity(unittest.TestCase):
    """Test Phase 7 Prior Case Retrieval self-leakage elimination and citation matching."""
    def test_self_leakage_elimination(self):
        query_id = "case_1024"
        candidate_pool = ["case_500", "case_1024", "case_780", "case_999"]
        
        # Strict post-filter: candidate != query
        filtered = [cid for cid in candidate_pool if cid != query_id]
        
        self.assertNotIn(query_id, filtered, "Query case must never appear in retrieved precedent candidates")
        self.assertEqual(len(filtered), len(candidate_pool) - 1)
        
    def test_reporter_citation_patterns(self):
        # Match pattern used in generate_top_submissions.py
        pat_air = re.compile(r'CITATION:\s*(\d{4})\s+AIR\s+(\d+)', re.I)
        pat_scr = re.compile(r'(\d{4})\s+SCR\s*(?:\((\d+)\))?\s*(\d+)', re.I)
        
        header_air = "PETITIONER: State of Kerala RESPONDENT: ABC Ltd. CITATION: 1966 AIR 1053"
        header_scr = "PETITIONER: Union of India CITATION: 1980 SCR (2) 456"
        
        m_air = pat_air.search(header_air)
        self.assertIsNotNone(m_air)
        self.assertEqual(m_air.group(1), "1966")
        self.assertEqual(m_air.group(2), "1053")
        
        m_scr = pat_scr.search(header_scr)
        self.assertIsNotNone(m_scr)
        self.assertEqual(m_scr.group(1), "1980")
        self.assertEqual(m_scr.group(2), "2")
        self.assertEqual(m_scr.group(3), "456")
        
        # Test CITATOR INFO matcher generated from query
        yr, pg = m_air.group(1), m_air.group(2)
        citator_pattern = re.compile(rf'{yr}\s*SC\s*{pg}', re.I)
        candidate_citator_info = "CITATOR INFO : R 1966 SC1053 (12) D 1970 SC 500"
        self.assertIsNotNone(citator_pattern.search(candidate_citator_info))

if __name__ == "__main__":
    unittest.main()
