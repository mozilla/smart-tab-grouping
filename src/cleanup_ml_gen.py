import pandas as pd

def remove_and(a: str):
    words = a.split(" ")
    if len(words) < 2:
        return a
    for stopword in ["and", "&"]:
        if stopword in words:
            and_pos = words.index(stopword)
            if and_pos > 0:
                return " ".join(words[0:and_pos])
    return " ".join(words)

def remove_or(a: str):
    words = a.split(" ")
    if len(words) < 3:
        return a
    if words[-2] == 'or':
        words = words[:-2]
    return " ".join(words)

def remove_year(a: str):
    words = list(filter(lambda word: word not in ["2023", "2024"], a.split(" ")))
    return " ".join(words)


def shorten_file(df):
    df["output_orig"] = df["output"]
    df["output"] = df["output"].apply(remove_and)
    df["output"] = df["output"].apply(remove_or)
    df["output"] = df["output"].apply(remove_year)
    return df

def cleanup(path):
    df = pd.read_csv(path)
    df = shorten_file(df)
    df.to_csv(f"{path}_updated")


if __name__ == "__main__":
    cleanup("../data/extract/topic_topic_fine_tuning_data__2025-02-21_16-50.csv")
    cleanup("../data/extract/topic_topic_fine_tuning_data__common_crawl_2025-02-23_08-18.csv")


