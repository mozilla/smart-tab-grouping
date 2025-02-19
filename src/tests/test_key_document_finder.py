import json
import pandas as pd
from util.key_document_finder import KeyDocumentFinder

def get_test_document(doc_name="pittsburgh_trip") -> pd.DataFrame:
   filename = f"./test_data/{doc_name}.json"
   with open(filename) as f:
      raw_data = json.load(f)
      tabs = raw_data["tab_list"]
      df = pd.DataFrame(tabs)

   documents_raw = (df["title"]).to_list()
   groups = df["windowId"].unique().tolist()
   topics = [groups.index(a) for a in df["windowId"].tolist()]
   test_doc = pd.DataFrame({"text_for_embedding": documents_raw, "ID": range(len(documents_raw)), "predicted_cluster": topics})
   return test_doc

def test_kewords():
   test_doc = get_test_document()
   kdf = KeyDocumentFinder(test_doc)
   kdf.compute_all()
   keywords_per_topic = kdf.keyword_list
   assert(keywords_per_topic[0][0] == "pittsburgh")
   assert(keywords_per_topic[1][0] == "latest")
   assert(keywords_per_topic[2][0] == "hotels")
   assert(keywords_per_topic[3][0] == "jira")
   assert(len(keywords_per_topic[0]) == 4)

# TODO - add test for Best Document selection
