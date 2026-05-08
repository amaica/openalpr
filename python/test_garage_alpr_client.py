#!/usr/bin/env python3
"""Testes leves para garage_alpr_client (unittest, sem pytest)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Repositório: .../openalpr/python/test_garage_alpr_client.py -> parent = python, parent.parent = raiz
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from garage_alpr_client import GarageAlprError, read_plate  # noqa: E402


class TestGarageAlprClient(unittest.TestCase):
    def test_missing_file(self) -> None:
        with self.assertRaises(GarageAlprError) as ctx:
            read_plate(REPO / "__nope__not_a_file__.png", repo_root=REPO)
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertIn("error", ctx.exception.payload)

    @unittest.skipUnless((REPO / "1.png").is_file(), "1.png not in repo")
    def test_1_png_success(self) -> None:
        d = read_plate(REPO / "1.png", repo_root=REPO)
        self.assertIn("plate", d)
        self.assertTrue(d["plate"])
        self.assertIn(d.get("used"), ("original", "enhanced"))
        self.assertIsInstance(d.get("confidence"), (int, float))

    @unittest.skipUnless((REPO / "2.png").is_file(), "2.png not in repo")
    def test_2_png_success(self) -> None:
        d = read_plate(REPO / "2.png", repo_root=REPO)
        self.assertIn("plate", d)
        self.assertTrue(d["plate"])
        self.assertIn(d.get("used"), ("original", "enhanced"))
        self.assertIsInstance(d.get("confidence"), (int, float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
