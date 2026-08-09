import csv
import tempfile
import unittest
from pathlib import Path

from topic_dataset.build_v2 import (
    assign_split,
    build_dataset,
    canonical_task_label,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TopicDatasetV2Test(unittest.TestCase):
    def test_canonical_task_labels_fit_product_contract(self):
        self.assertEqual(
            canonical_task_label("Reading Articles about Mental Health"),
            "Mental Health",
        )
        self.assertEqual(
            canonical_task_label("Researching or Booking Travel for Athens, Greece"),
            "Athens",
        )
        self.assertEqual(
            canonical_task_label("Using web based work applications"),
            "Work Apps",
        )

    def test_split_is_stable(self):
        self.assertEqual(assign_split("a-window", 42), assign_split("a-window", 42))

    def test_builder_keeps_window_variants_in_one_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repo_data = root / "data"
            gcs = root / "gcs"
            output = root / "output"
            gcs.mkdir()

            synthetic_rows = []
            for task_id, task, titles in (
                ("task-a", "Reading Articles about Space", ["Mars", "Moon", "Stars", "Rockets"]),
                ("task-b", "Online Shopping for Shoes", ["Running Shoes", "Boots"]),
            ):
                for title in titles:
                    synthetic_rows.append(
                        {
                            "url": "https://example.com",
                            "title": title,
                            "task": task,
                            "test_set_id": "window-1",
                            "task_id": task_id,
                            "user_description": "fixture",
                        }
                    )
            write_csv(
                repo_data / "synthetic_datasets" / "gen_test_set_fixture.csv",
                ["url", "title", "task", "test_set_id", "task_id", "user_description"],
                synthetic_rows,
            )
            write_csv(
                gcs / "single_tab_validation.csv",
                ["input_titles", "input_keywords", "output"],
                [{"input_titles": "Cheese", "input_keywords": "", "output": "Groceries"}],
            )
            write_csv(
                gcs / "fine_tuning_data__with_none__common_crawl_2025-02-23_08-18.csv",
                ["input_titles", "output"],
                [
                    {"input_titles": "Unclear page", "output": "None"},
                    {"input_titles": "Explicit fixture", "output": "Adult Movies"},
                    {"input_titles": "Ordinary store", "output": "Shopping"},
                ],
            )

            examples, audit = build_dataset(repo_data, gcs, output)

            self.assertEqual(audit["base_group_split_leaks"], [])
            self.assertEqual(audit["split_key_leaks"], [])
            window_rows = [row for row in examples if "window-1" in row.split_key]
            self.assertEqual(len({row.split for row in window_rows}), 1)
            self.assertTrue(any(row.output == "None" and row.tab_count == 2 for row in window_rows))
            safety_rows = [
                row for row in examples if row.source == "trust_safety_single"
            ]
            self.assertEqual({row.output for row in safety_rows}, {"None", "Adult Content"})
            self.assertEqual(audit["trust_safety_examples"], 2)
            self.assertTrue(all(row.input_text.startswith("Topic from keywords:") for row in examples))
            self.assertTrue((output / "train.csv").exists())
            self.assertTrue((output / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
