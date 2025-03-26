from util.shorten_topic_length import ShortenTopicLength
import pandas as pd


def test_shorten():
    test_df = pd.DataFrame(
        {"output": ["Desk concert concert", "dogs dogs dogs"], "input_titles": ["Desk concert", "dogs"],
         "input_keywords": [" desk concert", "bla"]})
    stl = ShortenTopicLength()
    df = stl.shorten_topics(test_df)
    assert df.iloc[0]["output"] == "Desk concert"
    assert df.iloc[1]["output"] == "dogs"