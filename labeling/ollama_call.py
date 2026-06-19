# import os
# import json
# from random import random
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import ollama
# import time

# # ==============================
# # CONFIG
# # ==============================

# MODEL_NAME = "gpt-oss:20b-cloud"
# OLLAMA_HOST = "http://localhost:11434"

# # Sentence-level concurrency (trong 1 file)
# SENTENCE_WORKERS = 3

# # File-level concurrency
# FILE_WORKERS = 2


# # ==============================
# # LLM CLASS
# # ==============================

# class OllamaLLM:
#     def __init__(self, model_name=MODEL_NAME, host=OLLAMA_HOST):
#         self.model_name = model_name
#         self.client = ollama.Client(host=host)

#     def get_response(self, prompt):
#         result = self.client.generate(
#             model=self.model_name,
#             prompt=prompt,
#             stream=False,
#         )
#         return result.get("response", "")


# # ==============================
# # PROMPT TEMPLATE
# # ==============================

# SYSTEM_PROMPT = '''
#         # Role: Aspect Category Identifier 
#         # Description:
#         You are an expert in identifying aspect categories in hotel reviews. You will be provided a review text and a list of words extracted from the review, and then check which category each word belongs to.
#         There are 6 categories to choose from:
#         - "Facility": Includes factors such as facilities, room furniture, decoration, pool area, etc.
#         - "Amenity": Includes public services such as parking, wifi, nearby attractions, security, etc.
#         - "Service": Includes service-related factors such as staff behavior, food quality, room service, check-in/check-out process, etc.
#         - "Experience": Includes overall experience factors such as value for money, comfort, ambiance, etc.
#         - "Branding": Overall satisfaction of customer compared to brand expectations.
#         - "Loyalty": Customer's willingness to return or recommend the hotel.
        
#         # Contraints:
#         - Your output will be in a JSON with key is each word from input list, and value is the corresponding aspect category.
#         - Only return a JSON, no need any other explanations or comments.
        
#         # Output:
#         A JSON file
        
#         # Input:
        
#         '''


# # ==============================
# # PROCESS FUNCTIONS
# # ==============================

# def process_sentence(ollama_call, sentence, word_list):

#     user_prompt = json.dumps(word_list, ensure_ascii=False)
#     big_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

#     response = ollama_call.get_response(big_prompt)
#     response = response.replace("```", "").replace("json", "").strip()

#     try:
#         return sentence, json.loads(response)
#     except:
#         return sentence, {}


# def _process_input_for_extract_category(sample):
#     result = sample["triplet_extraction"]
#     out = []

#     for sentence, value in result.items():
#         list_word = [entry["Aspect"] for entry in value]
#         out.append({
#             "sentence": sentence,
#             "list": list_word if list_word else []
#         })

#     return out


# def _process_output_for_extract_category(aspect_category_with_sentence, sample):
#     result = sample["triplet_extraction"]

#     for sentence, value in result.items():
#         aspect_category = aspect_category_with_sentence.get(sentence, {})
#         for entry in value:
#             aspect = entry["Aspect"]
#             entry["Category"] = aspect_category.get(aspect, "Unknown")

#     sample["four_extraction"] = result
#     return sample


# def extract_aspect_categories(ollama_call, inp_path):

#     # ===== Tạo output path =====
#     out_folder_path = os.path.dirname(inp_path).replace(
#         "triplet_batch",
#         "four_batch"
#     )
#     os.makedirs(out_folder_path, exist_ok=True)

#     output_path = os.path.join(
#         out_folder_path,
#         os.path.basename(inp_path)
#     )

#     # ===== SKIP nếu file đã tồn tại và hợp lệ =====
#     if os.path.exists(output_path):
#         try:
#             with open(output_path, "r", encoding="utf-8") as f:
#                 json.load(f)
#             return
#         except:
#             print(f"Reprocessing corrupted file: {os.path.basename(inp_path)}")

#     # ===== Load input =====
#     with open(inp_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     inputs = _process_input_for_extract_category(data)

#     aspect_category_with_sentence = {}

#     # ===== Sentence-level parallelism =====
#     with ThreadPoolExecutor(max_workers=SENTENCE_WORKERS) as executor:
#         futures = [
#             executor.submit(
#                 process_sentence,
#                 ollama_call,
#                 inp["sentence"],
#                 inp["list"]
#             )
#             for inp in inputs
#         ]

#         for future in as_completed(futures):
#             sentence, result = future.result()
#             aspect_category_with_sentence[sentence] = result

#     # ===== Merge output =====
#     new_entry = _process_output_for_extract_category(
#         aspect_category_with_sentence,
#         data
#     )

#     del new_entry["triplet_extraction"]

#     # ===== Save output =====
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(new_entry, f, indent=4, ensure_ascii=False)

#     print(f"Processed: {os.path.basename(inp_path)}")


# def process_file(file, base_path, ollama_call):
#     inp_path = os.path.join(base_path, file)
#     extract_aspect_categories(ollama_call, inp_path)


# # ==============================
# # MAIN
# # ==============================

# if __name__ == "__main__":

#     ollama_call = OllamaLLM()

#     base_path = "D:\\MABSA\\source\\data\\triplet_batch"
#     files = os.listdir(base_path)

#     # File-level parallelism
#     with ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
#         list(tqdm(
#             executor.map(
#                 lambda f: process_file(f, base_path, ollama_call),
#                 files
#             ),
#             total=len(files)
#         ))


import os
from tqdm import tqdm
import ollama
import json

class OllamaLLM:
    def __init__(self, model_name="gemma3:4b", host="http://127.0.0.1:3128"):
        
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)
        self.client.pull(self.model_name)

        
    def get_response(self, user_prompt, system_prompt: str = ""):
        # Combine system and user prompts into a single string
        big_prompt = system_prompt + "\n" + user_prompt
        
        result = self.client.generate(
            model=self.model_name, prompt=big_prompt  # Pass as a string
        )
        
        # Replace direct key access with .get to avoid KeyError
        return result.get("response", "No response available")
    
    

    
def classify_aspect_categories(ollama_call, input, system_prompt=""):
    # if system_prompt == "":
    user_prompt = '''
        # Role: Aspect Category Identifier 
        # Description:
        You are an expert in identifying aspect categories in hotel reviews. You will be provided a review text and a list of words extracted from the review, and then check which category each word belongs to.
        There are 6 categories to choose from:
        - "Facility": Includes factors such as facilities, room furniture, decoration, pool area, etc.
        - "Amenity": Includes public services such as parking, wifi, nearby attractions, security, etc.
        - "Service": Includes service-related factors such as staff behavior, food quality, room service, check-in/check-out process, etc.
        - "Experience": Includes overall experience factors such as value for money, comfort, ambiance, etc.
        - "Branding": Overall satisfaction of customer compared to brand expectations.
        - "Loyalty": Customer's willingness to return or recommend the hotel.
        
        # Contraints:
        - Your output will be in a JSON with key is each word from input list, and value is the corresponding aspect category.
        - Only return a JSON, no need any other explanations or comments.
        
        # Output:
        A JSON file
        
        # Input:
        
        '''
        
    response = ollama_call.get_response(user_prompt + json.dumps(input), system_prompt=system_prompt)
    response = response.replace("`", "").replace("json", "")
    
        # Convert the response string to a JSON object
    try:
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError:
        return response
    
    
    
def _process_input_for_extract_category(sample):

    result = sample["triplet_extraction"]
    out = []
    for key, value in result.items():
        list_word = [entry["Aspect"] for entry in value]
        
        out.append({
            "sentence": key,
            "list": list_word if list_word else []
        })
    # print(out)  
    return out

def _process_output_for_extract_category(aspect_category_with_sentence, sample):

    result = sample["triplet_extraction"]
    for key, value in result.items():
        aspect_category = aspect_category_with_sentence.get(key, {})
        for entry in value:
            aspect = entry["Aspect"]
            try:
                entry["Category"] = aspect_category[aspect]
            except:
                entry["Category"] = "Unknown"
            
    sample["four_extraction"] = result
    return sample
    
    
    
def extract_aspect_categories(ollama_call: OllamaLLM, inp_path: str):
    with open(inp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Configure output folder    
    out_folder_path = os.path.dirname(inp_path).replace("triplet_batch", "four_batch")
    os.makedirs(out_folder_path, exist_ok=True)
    output_path = os.path.join(out_folder_path, os.path.basename(inp_path))
    if not os.path.exists(output_path):
        # Extract aspect categories
        inputs = _process_input_for_extract_category(data)
        aspect_category_with_sentence = {}
        for inp in inputs:
            response = classify_aspect_categories(ollama_call, inp["list"])
            aspect_category_with_sentence[inp["sentence"]] = response
            
        new_entry = _process_output_for_extract_category(aspect_category_with_sentence, data)
        del new_entry["triplet_extraction"]


        with open(os.path.join(out_folder_path, os.path.basename(inp_path)), "w", encoding="utf-8") as f:
            json.dump(new_entry, f, indent=4, ensure_ascii=False)
            
        print(f"\nProcessed file {inp_path} and saved output to {output_path}")
    
if __name__ == "__main__":
    ollama_call = OllamaLLM(host="http://localhost:11434", model_name="gpt-oss:20b-cloud")
    # with open("data/hotel_review_10k/new_hotel_review_10k_triplet_extracted_full_processed.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    
    
    import os
    files = os.listdir("D:\\MABSA\\source\\data\\triplet_batch")
    for file in tqdm(files):
        inp_path = os.path.join("D:\\MABSA\\source\\data\\triplet_batch", file)
        extract_aspect_categories(ollama_call, inp_path)