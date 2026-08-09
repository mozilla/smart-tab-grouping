import unittest

from topic_dataset.firefox_text import (
    create_model_input,
    extract_group_keywords,
    extract_keywords,
    tokenize,
)


class FirefoxTopicTextTest(unittest.TestCase):
    def test_tokenize_matches_firefox_ascii_and_stop_words(self):
        self.assertEqual(tokenize("The quick fox jumped in 2025."), ["quick", "fox", "jumped", ""])
        self.assertEqual(tokenize("Google search: café"), ["caf", ""])

    def test_extract_keywords_matches_firefox_single_document_fixture(self):
        corpus = [
            "Planning a trip to Boston. Boston duck tours. Music in Boston. "
            "Flights to Boston. Planning trip back after."
        ]
        self.assertEqual(
            extract_keywords(corpus, max_words=5),
            [["boston", "planning", "trip", "duck", "tours"]],
        )

    def test_extract_keywords_matches_firefox_multi_document_fixture(self):
        corpus = [
            "Planning a trip to Boston. Boston duck tours. Music in Boston. Flights to Boston",
            "Planning a trip to Brazil. Flights to Brazil. Beach tour. More Planning.",
            "Planning dinner tonight. Brussel Sprouts. Meal Planning",
        ]
        self.assertEqual(
            extract_keywords(corpus),
            [
                ["boston", "duck", "tours"],
                ["brazil", "beach", "tour"],
                ["dinner", "tonight", "brussel"],
            ],
        )

    def test_group_keywords_use_first_three_titles(self):
        titles = ["Boston duck tours", "Boston flights", "Boston music", "Brazil beach"]
        self.assertEqual(
            extract_group_keywords(titles, ["Dinner recipes"]),
            ["boston", "duck", "tours"],
        )

    def test_model_input_matches_firefox(self):
        self.assertEqual(
            create_model_input([], ["One", "Two"]),
            "Topic from keywords: titles: \nOne \nTwo",
        )
        self.assertEqual(
            create_model_input(["boston", "travel"], ["One", "Two"]),
            "Topic from keywords: boston, travel. titles: \nOne \nTwo",
        )


if __name__ == "__main__":
    unittest.main()
