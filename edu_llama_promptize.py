"""
Reference Docs
[1] https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe
[2] https://js.langchain.com/docs/modules/data_connection/document_transformers/
[3] https://docs.pinecone.io/docs/overview
[4] https://python.langchain.com/docs/integrations/vectorstores/faiss
[5] https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
[6] https://cookbook.openai.com/examples/batch_processing
[7] https://platform.openai.com/docs/guides/batch/getting-started
"""
# %%
import pandas as pd
import json
from tqdm import tqdm
import os.path as osp


### [Step 1] Load Data and Make Loader ##embeddings as unique passage, question, paragraphs
def load_solvook_data(args):
    # load vector db
    with open(args.vector_db_path, "r") as f:
        data_dict = json.load(f)

    # load query
    query_db = pd.read_csv(args.query_path)
    for i in range(len(query_db)):
        query_db.loc[i, 'query'] = f'"Passage" : "{query_db.loc[i, "passage"]}", "Question" : "{query_db.loc[i, "question"]}"'
    
    if args.task == 2:
        query_db = query_db[query_db['relation']!=0].reset_index()

    return data_dict, query_db

    
    
## [Step 2] generation
def generation(args, retriever_dict, query_db):
    print("Start making prompts with top-k contents")
    query_list = list()
    for idx in tqdm(range(len(query_db))):
        #### Top-K search -----------------------------------------------------------------------------------------
        top_ques = retriever_dict['ques_db_retriever'].invoke(query_db['question'][idx])
        if args.task in [1,2]:
            top_mt = retriever_dict['mt_db_retriever'].invoke(query_db['passage'][idx])
            top_parap = retriever_dict['parap_db_retriever'].invoke(query_db['passage'][idx])
        
        top = top_ques
        if args.task in [1,2]:
            top += top_mt + top_parap
        
        ## Get pair
        top_content = list(); top_metadata = list()
        query_ = dict()
        for k in range(len(top)):
            top_content_ = f"["
            if args.task in [1, 2]:
                top_content_ = f"'Paragraph id': '{top[k].metadata['textbook_id']}_{top[k].metadata['unit_id']}_{top[k].metadata['story_id']}_{top[k].metadata['paragraph_id']}'. "

                try:
                    try:
                        top_content_ += f"'Paragraph': '{top[k].metadata['paragraphs']}'. "            
                    except:
                        top_content_ += f"'Paragraph': '{top[k].page_content}'. "
                except:
                    pass
            
                try:
                    top_content_ += f"'Passage': '{top[k].metadata['Passage']}'. "
                except:
                    top_content_ += f"'Passage': '{top[k].page_content}'. "
                
                try:
                    if top[k].metadata['relation'] != 0:
                        top_content_ += f" 'Relation': '{top[k].metadata['relation']}.'" 
                except:
                    pass
                
            elif args.task in [3, 4]:
                try:
                    top_content_ += f"'Question': '{top[k].metadata['question']}'. "
                except:
                    top_content_ += f"'Question': '{top[k].page_content}'. "
                
                top_content_ += f"'skill': '{top[k].metadata['skill']}'. 'method': '{top[k].metadata['method']}.'"
                
            top_content_ += "]"
            
            top_content.append(top_content_)
            
        top_content = '\n'.join(top_content)
        #### ---------------------------------------------------------------------------------------------------
        
        #### Make prompts --------------------------------------------------------------------------------------
        if args.task == 1:
            # paragraph
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Looking at the 'Passage’ and the ‘Question’, choose which of the ‘Candidates’ below is most relevant to the 'Paragraph' and answer the ‘Paragraph ID’ of it. (In this case, the ID will be something like 1_1_1_1)"
            query_['input'] = f"'Passage' : {query_db['passage'][idx]}. 'Question' : {query_db['question'][idx]}."
            query_['input'] += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += "\n Answer in form of <Paragraph>Paragraph id<Paragraph>. (For example, <Paragraph>1_1_1_1<Paragraph>)"
            query_['output'] = f"{str(query_db['textbook_id'][idx])}_{str(query_db['unit_id'][idx])}_{str(query_db['story_id'][idx])}_{str(query_db['paragraph_id'][idx])}"
        elif args.task == 2:
            # relation
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Looking at the 'Passage' and the 'Question', choose which of the 'Candidates' below is the most relevant to the 'Paragraph' and choose the relationship in 'Options'."
            query_['input'] = f"'Passage' : {query_db['passge'][idx]}. 'Question' : {query_db['question'][idx]}."
            query_['input'] += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += "'Options' : [1: Original (excerpting a part of the text without any changes or using the whole text as a passage), 2. Delete (deleting certain words/sentences from the text and using them as a passage), 3. Insert (adding words/sentences that were not in the text and using them as a passage), 4. Compound (a combination of original, deletion and insertion)]."
            query_['input'] += "\n Answer in form of <Relation>Answer(int)<Relation>(For example, <Relation>1<Relation>), and Provide a detailed description of why you chose the relationship in the form of a <Description>Reason for choosing the relationship information<Description>."
            query_['output'] = str(query_db['relation'][idx])
        elif args.task == 3:
            # skill
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>See ‘Candidates’ to view the ‘Question’ and select one of the ‘Options’ for your ability to solve this question."
            query_['input'] = f"'Question' : {query_db['question'][idx]}"
            query_['input'] += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += f"'Options' : [101: Understand the meaning of vocabulary (Understand the meaning of vocabulary), 102: Solve in English (Understand the meaning of vocabulary in English), 103: Mixed vocabulary, 201: Understand usage (Understand the usage), 202: Determine agreement or disagreement (Determine whether the usage is the same or different), 203: Mixed grammar, 301: Understand purpose (Understand the purpose of the text), 302: Understand topic (Understand the topic of the text. ), 303: Understand the title (Understand the title of a text), 304: Understand the claim (Understand the claim of a text), 305: Understand the gist (Understand the gist of a text), 306: Understand the meaning (Understand the meaning of a text), 307: Understand the mood (Understand the mood of a text), 308: Understand the mood of the speaker (Understand the mood of the speaker), 309: Understand the mood changes (Understand the mood changes of the speaker), 310: Understand the tone (Understand the tone of a text), 310: Understand the mood changes of a text. ), 310: Understanding Tone (Understand the tone of a text), 311: Understanding Order (Understand the order of a text), 312: Understanding the Object (Understand the object being referred to), 313: Mixing Content Understanding, 401: Inferring Content (Infer the content of a text), 402: Inferring Order (Infer the order of a text), 403: Inferring Lexicality (Infer the lexicality of a text), 404: Inferring Linking Words (Infer the linking words of a text), 405: Inferring Reference (Infer the reference of a text). ), 405: Referential inference (infer the referent), 406: Overall lexical inference (when asked for a combination of inferences), 407: Content matching (determine whether the content is the same or different), 408: Summarise (summarise the text), 409: Translation (translate the text into Korean), 410: English composition (write the text in English), 411: Mixed content application, 501: Domain integration, 601: Others]"
            query_['input'] += f"\nAnswer in form of <Skill>Answer(int)<Skill>. (For example, <Skill>405<Skill>)"
            query_['output'] = str(query_db['skill'][idx])
        elif args.task == 4:
            # method
            query_['instruction'] = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>See ‘Candidates’ and select one of the ‘Options’ options to see how the question asks to validate the learner's competency."
            query_['input'] = f"'Question' : {query_db['question'][idx]}"
            query_['input'] += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            query_['input'] += f"'Options' : [1: Find the correct one (singular) (find the correct one), 2: Find the correct one (plural) (find all the correct ones), 3: Count the correct ones (count) (find the correct ones and count them), 4: Find the incorrect one (singular) (find the incorrect one. ), 5: Find the wrong ones (plural) (Find all the wrong ones), 6: Count the wrong ones (count) (Find the wrong ones and count them), 7: Find something else (find something else), 8: Find the right position (find the right position), 9: Find the right arrangement (find the right arrangement). ), 9: Find the right arrangement (Find the right arrangement.), 10: Find the right combination (Find the right combination.), 11: Write the vocabulary (Choose from the view) (Find the correct vocabulary in the view and write it.), 12: Write the vocabulary (Find in the text) (Find the correct vocabulary in the text and write it. ), 13: Write vocabulary (correct/direct) (Correct or write with the correct vocabulary), 14: Write a sentence (Write a sentence), 15: Write in the correct arrangement (Write in the correct arrangement), 16: Mixed, 17: Other]"
            query_['input'] += f"\nAnswer in form of <Method>Answer(int)<Method>. (For example, <Method>5<Method>)"
            query_['output'] = str(query_db['method'][idx])
    
        query_list.append(query_)
                                

    return query_list



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split',type=str, default="tr", choices=['tr', 'val' ,'te'])
    
    parser.add_argument('--vector_db_path', type=str, default="./data/vector_db.json")
    parser.add_argument('--query_path', type=str, default="./data/solvook_handout_te.csv")
    
    parser.add_argument('--chunk_size', type=int, default=8000)
    parser.add_argument('--chunk_overlap', type=int, default=200)
    
    parser.add_argument('--embedding_model', type=str, default="text-embedding-3-small")
    
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], default=1,)

    args = parser.parse_args()
    
    if args.task == 1:
        task_name = 'paragraph'
        K = 6
    elif args.task == 2:
        task_name = 'relation'
        K = 6
    elif args.task == 3:
        task_name = 'skill'
        K = 3
    elif args.task == 4:
        task_name = 'method'
        K = 3

    if args.split == 'tr':
        args.query_path = args.db_path
    
    ### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
    print("[Step 1] Load Data!!")
    vector_db_dict, query_db = load_solvook_data(args)
    
    
    print("[Step 2] Start generation...")
    query_list = generation(args, vector_db_dict, query_db)
    
    with open(f'./{task_name}_top{K}_{args.split}.jsonl', 'w') as file:
        for query in query_list:
            file.write(json.dumps(query, ensure_ascii=False) + '\n')
    print("Finally End generation!!")
    