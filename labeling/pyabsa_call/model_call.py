import re
from pyabsa import AspectSentimentTripletExtraction as ASTE
import json
from tqdm import tqdm
import time
from typing import Dict

import os
import sys
import contextlib
import re

# find a suitable checkpoint and use the name:
triplet_extractor = ASTE.AspectSentimentTripletExtractor(
    checkpoint="english"
)  # here I use the english checkpoint which is trained on all English datasets in PyABSA


def split_sentences(review: str):
    review = re.sub(r'\b(Mr|Ms|Miss)\.', r'\1', review)
    sentence_split = [s for s in review.split(".") if len(s.strip()) > 5]
    return sentence_split

def remove_metadata(sample:Dict):
    temp_review_photo = sample.get("review_photo", {})
    
    keys_to_remove = ['name', 'country', 'room', 'state', 'review_score', 'review_date', "review_title", 'date']
    for key in keys_to_remove:
        if key in sample:
            del sample[key]
    # handle image
    new_review_photo = {str(i): key for i, key in enumerate(temp_review_photo.keys()) if key is not None}
    sample["review_photo"] = new_review_photo

    return sample


def extract_triplets(extractor, data_path="../../data/hotel_review_10k_only_merge.json", start_idx=0, end_idx=100):
    out_folder_path = os.path.join(os.path.dirname(data_path), "data", "triplet_batch")
    os.makedirs(out_folder_path, exist_ok=True)
    triplet_extractor = extractor

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)
    
    for i, entry in tqdm(enumerate(data), total=len(data)):
        if i < start_idx:
            continue
        if i >= end_idx:
            break
            
        # check file exists to skip
        id = entry.get("id", str(i))
        outpath = os.path.join(out_folder_path, f"{id}.json")
        # if os.path.exists(outpath):
        #     continue
        
        # else:
        # Extract triplets

        review = entry.get("review", "")    
        review = re.sub(r'\b(Mr|Ms)\.', r'\1', review)
        sentence_split = [
            s.strip() 
            for s in re.split(r'[.!?]+', review) 
            if len(s.strip()) > 5
        ]

        result = {}
        for s in sentence_split:
            s = s.strip()
            # store triplet extraction result
            temp_result = triplet_extractor.predict(s, print_result=False).get("Triplets", [])
            if type(temp_result) == list and len(temp_result) > 0:
                result[s] = temp_result
            else:
                result[s] = []

        
        entry["triplet_extraction"] = result
        entry = remove_metadata(entry)

        # save output
        with open(outpath, "w") as f:
            json.dump(entry, f, indent=4)




# def re_extract_triplets(extractor, retry=3,
#                         data_path="../../data/hotel_review_10k_triplet_extracted_full.json"):
#     triplet_extractor = extractor

#     # Load data
#     with open(data_path, "r") as f:
#         data = json.load(f)

#     for i, entry in tqdm(enumerate(data), total=len(data)):
#         results = entry.get("result", None)
#         if results:
#             for res in results:
#                 for key, value in res.items():
#                     if value == []:
#                         for r in range(retry):
#                             s = key.strip()
#                             new_res = triplet_extractor.predict(s, print_result=False)["Triplets"]
#                             if new_res:
#                                 results[key] = new_res
#                                 break

#     with open(
#         data_path,
#         "w",
#         ) as f:
#         json.dump(data, f, indent=4)

# def check_missing_triplets():
#     # Load data
#     with open("../../data/hotel_review_10k_triplet_extracted_full.json", "r") as f:
#         data = json.load(f)

#     missing_count = 0
#     missing_triplet_samples = []
#     for i, entry in tqdm(enumerate(data), total=len(data)):
#         results = entry.get("result", None)
#         if results:
#             for res in results:
#                 for key, value in res.items():
#                     word_count = len(key.split(" "))
#                     if value == [] and word_count < 2:
#                         missing_count += 1
#                         missing_triplet_samples.append(entry)
                        
#                         # print(f"Missing triplet in review {i}: '{entry}'")
#                         break
#     print(f"Total missing triplets: {missing_count}")

#     with open(
#         f"../../data/hotel_review_10k_missing_triplet_samples.json",
#         "w",
#         ) as f:
#         json.dump(missing_triplet_samples, f, indent=4)



if __name__ == "__main__":
    # check_missing_triplets()
    extract_triplets(triplet_extractor, data_path="D:\\MABSA\\source\\reviews_photo_english.json", start_idx=0)