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
import numpy as np
import re, os, json, time
from tqdm import tqdm
import os.path as osp

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




def extract_label(text):
    info = dict()
    try:
        try:
            paragraph = re.search(r'<Paragraph>(.*?)<Paragraph>', text).group(1).strip()
        except:
            try:
                paragraph = re.search(r'<Paragraph>(.*?)</Paragraph>', text).group(1).strip()
            except:
                paragraph = re.search(r'<Paragraph>(.*?)<\Paragraph>', text).group(1).strip()
        info['paragraph'] = paragraph
    except:
        print("Failed to extract the paragraph label")
        info['paragraph'] = -9999
        
    try:
        try:
            skill = re.search(r'<Skill>(.*?)<Skill>', text).group(1).strip()
        except:
            try:
                skill = re.search(r'<Skill>(.*?)</Skill>', text).group(1).strip()
            except:
                skill = re.search(r'<Skill>(.*?)<\Skill>', text).group(1).strip()
        info['skill'] = skill
    except:
        print("Failed to extract the skill label")
        info['skill'] = -9999
        
    try:
        try:
            method = re.search(r'<Method>(.*?)<Method>', text).group(1).strip()
        except:
            try:
                method = re.search(r'<Method>(.*?)</Method>', text).group(1).strip()
            except:
                method = re.search(r'<Method>(.*?)<\Method>', text).group(1).strip()
        info['method'] = method
    except:
        print("Failed to extract the method label")
        info['method'] = -9999
        
    try:
        try:
            relation = re.search(r'<Relation>(.*?)<Relation>', text).group(1).strip()
        except:
            try:
                relation = re.search(r'<Relation>(.*?)</Relation>', text).group(1).strip()
            except:
                relation = re.search(r'<Relation>(.*?)<\Relation>', text).group(1).strip()
        info['relation'] = relation
    except:
        print("Failed to extract the relation label")
        info['relation'] = -9999
        
    try:
        try:
            description = re.search(r'<Description>(.*?)<Description>', text).group(1).strip()
        except:
            try:
                description = re.search(r'<Description>(.*?)</Description>', text).group(1).strip()
            except:
                description = re.search(r'<Description>(.*?)<\Description>', text).group(1).strip()
        info['description'] = description
    except:
        print("Failed to extract the description label")
        info['description'] = -9999
    
    return info
    

    
    
## generation
def generation(args, retriever_dict, query_db):

    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key)
    
    top_content_list = list()
    tasks = list()
    print("Start making prompts with top-k contents to send them into batch query")
    idx_list = list()
    id_list = list(); textbook_list = list(); unit_list = list(); story_list = list(); paragraph_list = list()
    skill_list = list(); method_list = list(); relation_list = list(); description_list = list()
    answer_list = list()
    for idx in tqdm(range(len(query_db))):
        #############################################################################
        ## Top-K search
        #############################################################################
        if args.task in [1, 2]:
            top_mt = retriever_dict['mt_db_retriever'].invoke(query_db['passage'][idx])
            top_parap = retriever_dict['parap_db_retriever'].invoke(query_db['paragraph'][idx])
        top_ques = retriever_dict['ques_db_retriever'].invoke(query_db['question'][idx])      # 질문 v.s. 질문*
        
        top = top_ques
        if args.task in [1, 2]:
            top += top_mt + top_parap
        
        ## Get pair
        top_content = list()
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
                    top_content_ += f"'Passage': '{top[k].metadata['passage']}'. "
                except:
                    top_content_ += f"'Passage': '{top[k].page_content}'. "
                
                try:
                    if top[k].metadata['relation'] != 0:
                        top_content_ += f" '관계': '{top[k].metadata['relation']}.'" 
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
        top_content_list.append(top_content)
        #############################################################################
        #############################################################################
        
        
        #############################################################################    
        ### make every  -------------------------------------------------------------
        #############################################################################
        if args.task == 1:
            # paragraph
            sys_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Looking at the 'Passage’ and the ‘Question’, choose which of the ‘Candidates’ below is most relevant to the 'Paragraph' and answer the ‘Paragraph ID’ of it. (In this case, the ID will be something like 1_1_1_1)"
            user_prompt = f"'Passage' : {query_db['passage'][idx]}. 'Question' : {query_db['question'][idx]}."
            user_prompt += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            user_prompt += "\n Answer in form of <Paragraph>Paragraph id<Paragraph>. (For example, <Paragraph>1_1_1_1<Paragraph>)"
        elif args.task == 2:
            # relation
            sys_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Looking at the 'Passage' and the 'Question', choose which of the 'Candidates' below is the most relevant to the 'Paragraph' and choose the relationship in 'Options'."
            user_prompt = f"'Passage' : {query_db['passge'][idx]}. 'Question' : {query_db['question'][idx]}."
            user_prompt += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            user_prompt += "'Options' : [1: Original (excerpting a part of the text without any changes or using the whole text as a passage), 2. Delete (deleting certain words/sentences from the text and using them as a passage), 3. Insert (adding words/sentences that were not in the text and using them as a passage), 4. Compound (a combination of original, deletion and insertion)]."
            user_prompt += "\n Answer in form of <Relation>Answer(int)<Relation>(For example, <Relation>1<Relation>), and Provide a detailed description of why you chose the relationship in the form of a <Description>Reason for choosing the relationship information<Description>."
        elif args.task == 3:
            # skill
            sys_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>See ‘Candidates’ to view the ‘Question’ and select one of the ‘Options’ for your ability to solve this question."
            user_prompt = f"'Question' : {query_db['question'][idx]}"
            user_prompt += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            user_prompt += f"'Options' : [101: Understand the meaning of vocabulary (Understand the meaning of vocabulary), 102: Solve in English (Understand the meaning of vocabulary in English), 103: Mixed vocabulary, 201: Understand usage (Understand the usage), 202: Determine agreement or disagreement (Determine whether the usage is the same or different), 203: Mixed grammar, 301: Understand purpose (Understand the purpose of the text), 302: Understand topic (Understand the topic of the text. ), 303: Understand the title (Understand the title of a text), 304: Understand the claim (Understand the claim of a text), 305: Understand the gist (Understand the gist of a text), 306: Understand the meaning (Understand the meaning of a text), 307: Understand the mood (Understand the mood of a text), 308: Understand the mood of the speaker (Understand the mood of the speaker), 309: Understand the mood changes (Understand the mood changes of the speaker), 310: Understand the tone (Understand the tone of a text), 310: Understand the mood changes of a text. ), 310: Understanding Tone (Understand the tone of a text), 311: Understanding Order (Understand the order of a text), 312: Understanding the Object (Understand the object being referred to), 313: Mixing Content Understanding, 401: Inferring Content (Infer the content of a text), 402: Inferring Order (Infer the order of a text), 403: Inferring Lexicality (Infer the lexicality of a text), 404: Inferring Linking Words (Infer the linking words of a text), 405: Inferring Reference (Infer the reference of a text). ), 405: Referential inference (infer the referent), 406: Overall lexical inference (when asked for a combination of inferences), 407: Content matching (determine whether the content is the same or different), 408: Summarise (summarise the text), 409: Translation (translate the text into Korean), 410: English composition (write the text in English), 411: Mixed content application, 501: Domain integration, 601: Others]"
            user_prompt += f"\nAnswer in form of <Skill>Answer(int)<Skill>. (For example, <Skill>405<Skill>)"
        elif args.task == 4:
            # method
            sys_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>See ‘Candidates’ and select one of the ‘Options’ options to see how the question asks to validate the learner's competency."
            user_prompt = f"'Question' : {query_db['question'][idx]}"
            user_prompt += f"\n'Candidates' : {top_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            user_prompt += f"'Options' : [1: Find the correct one (singular) (find the correct one), 2: Find the correct one (plural) (find all the correct ones), 3: Count the correct ones (count) (find the correct ones and count them), 4: Find the incorrect one (singular) (find the incorrect one. ), 5: Find the wrong ones (plural) (Find all the wrong ones), 6: Count the wrong ones (count) (Find the wrong ones and count them), 7: Find something else (find something else), 8: Find the right position (find the right position), 9: Find the right arrangement (find the right arrangement). ), 9: Find the right arrangement (Find the right arrangement.), 10: Find the right combination (Find the right combination.), 11: Write the vocabulary (Choose from the view) (Find the correct vocabulary in the view and write it.), 12: Write the vocabulary (Find in the text) (Find the correct vocabulary in the text and write it. ), 13: Write vocabulary (correct/direct) (Correct or write with the correct vocabulary), 14: Write a sentence (Write a sentence), 15: Write in the correct arrangement (Write in the correct arrangement), 16: Mixed, 17: Other]"
            user_prompt += f"\nAnswer in form of <Method>Answer(int)<Method>. (For example, <Method>5<Method>)"
        #############################################################################
        #############################################################################
        
                                 
        task = {
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.llm_model,
                    "temperature": args.temperature,
                    ## incompatible with "gpt-4o"
                    # "response_format": { 
                    #     "type": "json_object"
                    # },
                    "messages": [
                        {
                            "role": "system",
                            "content": sys_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                }
            }
        tasks.append(task)
    
    ## create batch file (json type)
    file_name = osp.join(args.result_path, f"batch_tasks.jsonl")
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

    ## uploading batch file
    batch_file = client.files.create(
                file=open(file_name, "rb"),
                purpose="batch"
                )

    ## creating the batch job
    batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
                )
    print(f"Batch API job ID : {batch_job.id}")
        
    ## checking batch status
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch API status : {batch_job.status}")
        if batch_job.status in ['failed', "cancelling", "cancelled", "expired"]:
            raise ValueError("Fail to send batch api:(")
        elif batch_job.status in ["finalizing"]:
            print("Waiting for result being prepared")
        elif batch_job.status in ["completed"]:
            print(f"Completed batch job!")
            break
        time.sleep(30)
        
    ## retrieving results and save as jsonl file
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    result_file_name = osp.join(args.result_path, f"batch_job_results.jsonl")
    with open(result_file_name, 'wb') as file:
        file.write(result)
    
    with open(result_file_name, 'wb') as file:
        file.write(result)
        ## Loading data from saved file (json)
        results = []
        with open(result_file_name, 'r') as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.strip())
                results.append(json_object)
                    
        for i, res in enumerate(results):
            task_id = res['custom_id']
            # Getting index from task id
            index = task_id.split('-')[-1]
            
            # get index in query_db
            query_idx = query_db.iloc[int(index)]['handout_id']
            idx_list.append(query_idx)
            
            answer = res['response']['body']['choices'][0]['message']['content']
            # print(f"RESULT : batch:{batch}/{num_batches}. idx:{start_idx+i}. \n {answer}")
            print(f"RESULT {i} : {answer}")
            answer_list.append(answer)
            
            tmp_dict = extract_label(answer)
            id_list.append(tmp_dict['paragraph'])
            try:
                textbook_list.append(float(tmp_dict['paragraph'].split('_')[0]))
                unit_list.append(float(tmp_dict['paragraph'].split('_')[1]))
                story_list.append(float(tmp_dict['paragraph'].split('_')[2]))
                paragraph_list.append(float(tmp_dict['paragraph'].split('_')[3]))
            except:
                textbook_list.append(-9999)
                unit_list.append(-9999)
                story_list.append(-9999)
                paragraph_list.append(-9999)
                
            skill_list.append(tmp_dict['skill'])
            method_list.append(tmp_dict['method'])
            relation_list.append(tmp_dict['relation'])
            try:
                description_list.append(tmp_dict['description'])
            except:
                description_list.append(-9999)
            print("\n\n----------------------------\n\n")
        
        
    ## save_result
    try:
        label_df = pd.DataFrame({"query_idx" : idx_list,
                                "id" : id_list,
                                "textbook_id" : textbook_list,
                                "unit_id" : unit_list,
                                "story_id" : story_list,
                                "paragraph_id" : paragraph_list,
                                "skill" : skill_list,
                                "method" : method_list,
                                "relation" : relation_list,
                                "description" : description_list,
                                },)
        label_df.to_csv(osp.join(args.result_path, 'answer_df.csv'),
                                encoding="utf-8-sig", index=False)
    except:
        print("Problem with saving label df")

    with open(osp.join(args.result_path, 'answer_list.json'), 'w') as json_file:
        json.dump(answer_list, json_file, ensure_ascii=False)
        
    with open(osp.join(args.result_path, 'top_k_list.json'), 'w') as json_file:
        json.dump(top_content_list, json_file, ensure_ascii=False)
    
    print(f"Save result(answer and top_k) in {args.result_path}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--openai_api_key',type=str, default=None, required=True)
    
    parser.add_argument('--query_path', type=str, default=".")
    parser.add_argument('--vector_db_path', type=str, default=".")
    
    parser.add_argument('--llm_model', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.0)
    
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], default=1,
                        help="1: paragraph, 2: relation, 3: skill, 4: method")
    parser.add_argument('--in_context_sample', action='store_true', default=False,
                        help='in case of task no.2, adopt in-context sample, not retrieved sample')
        
    parser.add_argument('--result_path', type=str, default='./exp_result')
    
    parser.add_argument('--only_train', action='store_true', default=False)

    
    args = parser.parse_args()
    
    ## set save_path
    save_name = f"seed_{args.seed}/task_{args.task}/{args.llm_model}"
    args.result_path = osp.join(args.result_path, save_name)
    print(f"Set save path on {args.result_path}")
    os.makedirs(args.result_path, exist_ok=True)
    
    print("[Step 1] Load Data!!")
    vector_db, query_db = load_solvook_data(args)
    
    args_dict=vars(args)
    with open(osp.join(args.result_path, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, ensure_ascii=False)
    
    print("[Step 2] Start generation...")
    generation(args, vector_db, query_db)
    print("Finally End generation!!")