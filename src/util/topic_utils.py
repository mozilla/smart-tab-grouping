import random

import pandas as pd

from util.key_document_finder import KeyDocumentFinder

LEGACY_DOC_SELECTION = -1

def compute_topic_using_digest(topic_generator, key_doc_finder, items, group_item, num_keywords=LEGACY_DOC_SELECTION):
    important_doc_indices = key_doc_finder.get_best_documents()
    picked_documents = items.loc[important_doc_indices[group_item], "title"].tolist()[:3]
    keywords = key_doc_finder.get_keywords_for_group(group_item)
    # use first keyword only
    if num_keywords == LEGACY_DOC_SELECTION:
        all_docs = items.loc[important_doc_indices[group_item], "title"].tolist()
        random.shuffle(all_docs)
        cluster_data = {"documents": all_docs[:3]}
    else:
        cluster_data = {"documents": [" ".join(keywords[:num_keywords])] + picked_documents}
    topic = topic_generator.get_topic(cluster_data)
    return topic, picked_documents, keywords


def create_topic_training_dataset(df: pd.DataFrame, user_label_key: str, topic_generator, predicted_id_topics):
    user_topics = df[user_label_key].unique().tolist()
    key_doc_finder_user = KeyDocumentFinder(df, user_label_key, "title")
    key_doc_finder_user.compute_all()
    topic_trainers = []
    for user_topic in user_topics:
        if isinstance(user_topic, float):
            user_topic = int(user_topic)
        header_title = predicted_id_topics.get(str(user_topic), user_topic)
        pred_topic, picked_documents, keywords = compute_topic_using_digest(topic_generator, key_doc_finder_user, df,
                                                                            user_topic, num_keywords=1)
        legacy_topic, _x, _x = compute_topic_using_digest(topic_generator, key_doc_finder_user, df,
                                                                            user_topic, num_keywords=LEGACY_DOC_SELECTION)
        zero_keyword_topic, _x, _x = compute_topic_using_digest(topic_generator, key_doc_finder_user, df,
                                                                            user_topic, num_keywords=0)
        two_keyword_topic, _x, _x = compute_topic_using_digest(topic_generator, key_doc_finder_user, df,
                                                                            user_topic, num_keywords=2)

        topic_trainers.append({"label": header_title, "three_titles": "\n".join(picked_documents), "keywords": ",".join(keywords),
                   "ai_pred_topic": pred_topic, "ai_pred_topic_2kw": two_keyword_topic, "ai_pred_topic_0kw": zero_keyword_topic,
                               "ai_pred_topic_legacy": legacy_topic})

    return pd.DataFrame(topic_trainers)
