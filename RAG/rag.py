import psutil
import os
import random
from collections import Counter
import time
import logging
from util.logger_config import logger
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import lark
from langchain_community.document_loaders import JSONLoader
import json
import time
import tiktoken
from langchain_core.messages import AIMessage
import re
from typing import Optional, List


class RAG():
    def __init__(self, dbms, benchmark, vector_store_path):
        self.dbms = dbms
        self.benchmark = benchmark
        self.vector_stor_path = vector_store_path
        self.cpu_cores, self.ram_size, self.disk_size = self.get_hardware_info()
        self.disk_type = self.get_disk_type()
        self.worklod_info = self.get_wokload_info()
        self.data_info = self.get_data_info()
        self.money = 0
        self.token = 0
        self.input_token = 0
        self.output_token = 0
        self.model_name = os.getenv("LLM_MODEL")
        self.cost_path = f"optimization_results/{self.dbms.name}/costs.txt"

    def calc_token(self, in_text, out_text=""):
        enc = tiktoken.encoding_for_model(self.model_name)
        self.input_token +=  len(enc.encode(in_text))
        self.output_token += len(enc.encode(out_text))
        self.token += len(enc.encode(in_text + out_text))
    
    def add_tokens(self, message):
        logger.info(message)
        logger.info(message.usage_metadata)
        self.token += message.usage_metadata['total_tokens']
        self.input_token += message.usage_metadata['input_tokens']
        self.output_token += message.usage_metadata['output_tokens']
    
    def calc_money(self, in_token, out_token):
        """money for gpt4"""
        if self.model_name == "gpt-4":
            return (in_token * 0.03 + out_token * 0.06) / 1000
        elif self.model_name == "gpt-3.5-turbo":
            return (in_token * 0.0015 + out_token * 0.002) / 1000
        elif self.model_name == "gpt-4o-mini":
            return (in_token * 0.00015 + out_token * 0.0006) / 1000
    


    def extract_json(self,message: AIMessage) -> List[dict]:
        """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

        Parameters:
            text (str): The text containing the JSON content.

        Returns:
            list: A list of extracted JSON strings.
        """
        text = message.content


        self.add_tokens(message)
        # Define the regular expression pattern to match JSON blocks
        pattern = r"```json(.*?)```"

        # Find all non-overlapping matches of the pattern in the string
        matches = re.findall(pattern, text, re.DOTALL)
        logger.debug(matches)

        # Return the list of matched JSON strings, stripping any leading or trailing whitespace
        try:
            return [json.loads(match.strip()) for match in matches]
        except Exception:
            raise ValueError(f"Failed to parse: {message}")
        
    def get_hardware_info(self):
        available_cpu_cores = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        total_memory = memory.total
        total_memory = total_memory / (1024 * 1024 * 1024)
        root_disk = psutil.disk_usage('/')
        total_disk_space = root_disk.total
        total_disk_space = total_disk_space / (1024 * 1024 * 1024)
        return available_cpu_cores, int(total_memory), int(total_disk_space)
    
    def get_disk_type(self, device="sda"):
        rotational_path = f'/sys/block/{device}/queue/rotational'
        if os.path.exists(rotational_path):
            with open(rotational_path, 'r') as file:
                rotational_value = file.read().strip()
                if rotational_value == '0':
                    return 'SSD'
                elif rotational_value == '1':
                    return 'HDD'
                else:
                    return 'Unknown'
        else:
            return 'Unknown'
        
    def get_wokload_info(self):
        workload_config_path = f"./benchbase/target/benchbase-{self.dbms.name}/config/{self.dbms.name}/sample_{self.benchmark}_config.xml"
        olap = ["tpch"]
        des = ""
        import xml.etree.ElementTree as ET
        tree = ET.parse(workload_config_path)
        root = tree.getroot()
        

        # Extracting specific elements
        terminals = root.find('terminals').text
        
        if self.benchmark in olap:
            workload_type = "OLAP"
            combined_info = f"{workload_type}, {self.benchmark}"

        else:
            workload_type = "OLTP"
            weight = root.find('./works/work/weights').text
            weights = weight.split(',')
            # Extract transaction types
            transactiontypes = root.findall('./transactiontypes/transactiontype')
            transaction_names = [txn.find('name').text for txn in transactiontypes]
            combined_info = f"{workload_type}, {self.benchmark}, Terminals: {terminals}, Weights: {','.join(weights)}, Transaction Types: {', '.join(transaction_names)}"
        return combined_info
    
    def get_data_info(self):
        db_size = self.dbms.get_data_size()
        tables_info = self.dbms.get_tables_info()
        return f"{db_size} MB data, the table name and the rows of every table: {tables_info}"

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["knob_name"] = record.get("name")
        metadata["version"] = "13"
        return metadata
        
    # Persist vector_store
    def persist(self, knowledge_folder_path, vector_store_path): 
        document_list = []
        for filename in os.listdir(knowledge_folder_path):
            file_path_name = os.path.join(knowledge_folder_path,filename)
            # file_path_name = "./postgres/system_view.json"
            logger.info("[Loading JSON] %s",file_path_name)
            loader = JSONLoader(
                file_path=file_path_name,
                jq_schema=".params[]",
                text_content=False,
                metadata_func = self.metadata_func
            )
            documents = loader.load()
            document_list.extend(documents)
        logger.info(len(document_list))
        random.shuffle(document_list)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        # documents = text_splitter.split_documents(document)
        print("start ")
        embeddings = OpenAIEmbeddings()
       
        # persist
        length = 50
        num = len(document_list) // length
        remainder = len(document_list) % length
        for i in range(num):
            start_index = i * length
            end_index = start_index + length
            sublist = document_list[start_index: end_index]
            docsearch = Chroma.from_documents(sublist, embeddings, persist_directory=vector_store_path)
            docsearch.persist()
            print(f"persist {num}" )
            time.sleep(20)

        if remainder > 0:
            sublist = document_list[-remainder:]
            docsearch = Chroma.from_documents(sublist, embeddings, persist_directory=vector_store_path)
            docsearch.persist()
    
    def get_answer(self, system_prompt, question_list):
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        metadata_field_info = [
            AttributeInfo(
                name="knod_name",
                description="The name of knob which can be tuned in DBMS.",
                type="string"
            )
        ]
        document_content_description = "The information of knobs"
        llm = ChatOpenAI(model=self.model_name,temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm,
            docsearch,
            document_content_description,
            metadata_field_info,
        )
        # retriever = docsearch.as_retriever(search_kwargs={"k": 5})
        system_template =  system_prompt + """
        --------------------------------------------------------
        <chat_history>
        {chat_history}
        </chat_history>
        ---------------------------------------------------------
        """

        human_prompt = """
        <question>
        {question}
        </question>
        """

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model=self.model_name,temperature=0.1),
            retriever,
            condense_question_prompt=prompt,
            return_source_documents=True,
        )

        all_data_list = []
        for question in question_list:
            time.sleep(5)
            chat_history = []
            result = qa.invoke({"question": question, "chat_history": chat_history})
            chat_history.append((question, result["answer"]))
            logger.info("[Related Knowledge] %s", result["source_documents"][0])
            logger.info("[Question]：%s", question)
            logger.info("[Response]：%s", result["answer"])
            all_data_list.extend(json.loads('['+result["answer"]+']'))
        return all_data_list
    

