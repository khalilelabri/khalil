from stchatgpt import ask_st_chatgpt
import json
import os 
import csv

def generate_text_summary(text):
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text. \
    Give a concise summary of the text that is well optimized for retrieval. 
    Text: {} """.format(text)

    write_message(prompt_text)
    send_prompt()
    summarized_text=retrieve_message()
    return summarized_text

def generate_text_summaries(json_file_path):
    with open(json_file_path,'r') as file:
        dictionary = json.load(file)
    summarized_texts = dict()
    for title in dictionary:
        summarized_texts[title] = generate_text_summary(dictionary[title])
    return summarized_texts

def generate_table_summary(table):
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing table for retrieval. \
    These summaries will be embedded and used to retrieve table elements. \
    Give a concise summary of the table ,provided in csv format, that is well optimized for retrieval. 
    Table: {} """.format(table)

    write_message(prompt_text)
    send_prompt()
    summarized_text=retrieve_message()
    return summarized_text

def generate_table_summaries(folder_path):
    for csv_file in os.listdir(folder_path):
        with open(csv_file_path,'r') as file:
            dictionary = json.load(file)
        summarized_tables = dict()
        for title in dictionary:
            summarized_tables[title] = generate_text_summary(dictionary[title])
    return summarized_texts


