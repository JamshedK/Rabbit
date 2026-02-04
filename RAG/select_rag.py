from RAG.rag import RAG
from util.logger_config import logger
import textwrap
from langchain_core.tools import tool
from RAG.grah_rag import search_local, search_global
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from typing import Optional, List
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.messages import AIMessage
import re
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# class Knob(BaseModel):
#     name: str = Field(..., description="The name of knob in KNOB COLLECTION")
#     reson: str = Field(...,description="The reson why the knob in KNOB COLLECTION is depend on the other knobs")
#     interdependent_knobs: str = Field(..., description="The name of interdependent knobs. If no interdependent knobs, set it with 'None'")

# class Answer(BaseModel):
#     Knobs: List[Knob]
# class Answer(BaseModel):
#     knob_names: str= Field(..., description="The names of the suggested important knobs which are in the current dbms, but not in the knob collection list. Multiple knob names are separated by ','")

class SelectRAG(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, version, target_knobs=None, candidate_knobs=None):
        super().__init__(dbms, benchmark, vector_store_path)
        self.target_knobs = target_knobs # selected knobs from similar historical tasks
        self.version = version
        self.candidate_knobs = candidate_knobs

    
    
    def load_data(self):
        """
        返回按照重要性排序的旋钮
        """
        system_prompt = textwrap.dedent(f"""
        Suppose you are an experienced DBA, and you are required to tune the knobs of {self.dbms.name}.
        "Use the following pieces of retrieved context to answer "
            "the question. "
            "\n\n"
            "{{context}}"
        """) 
        question = textwrap.dedent(f"""
    
        Candidate Knobs:{self.candidate_knobs}
        DBMS:{self.dbms.name}_{self.version}
        WORKLOAD: {self.worklod_info}
        DATA: {self.data_info}
        Hardware Information: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
        TASK:
        Your will determine which knobs are worth tuning. You only tune knobs that have a significant impact on DBMS performance. 
        Given the following candidate knobs, score the importance for each knob between 0 and 1, with a higher value indicating that it is more likey to impact {self.dbms.name} performance significantly. 
        Which knobs are important heavily depends on the workload, data information, and hardware environment.
        Now let us think step by step and give me your scoring of all the candidate knobs in json format:
        {{
            "knob_name": {{score}}    // fill "score" with a number between 0 and 1
        }}
        
        
        """
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        llm = ChatOpenAI(model=self.model_name)
        embeddings = OpenAIEmbeddings()

        # retriever
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        metadata_field_info = [
            AttributeInfo(
                name="knob_name",
                description="The name of knob which can be tuned in DBMS.",
                type="string"
            )
            # TODO add version
        ]
        document_content_description = "The information of knobs"
        retriever = SelfQueryRetriever.from_llm(
            llm,
            docsearch,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k": 5}
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        prompt_llm = create_retrieval_chain(retriever, question_answer_chain)
        
        # prompt_llm = prompt | llm | self.extract_json
        all_data = prompt_llm.invoke({"input":question})

        # return the cost
        self.calc_token(all_data['input']+ ''.join(str(item) for item in all_data['context']), all_data['answer'])
        with open(self.cost_path, "a") as f:
            f.write("Select_RAG: \n")
            f.write("Total Tokens: "+ str(self.token))
            f.write("\n")
            f.write("Input Tokens: "+ str(self.input_token)+ ". Output Tokens: " + str(self.output_token))
            f.write("\n")
            f.write("Cost: "+ str(self.calc_money(self.input_token, self.output_token)))
            f.write("\n \n")

        knobs_score = self.extract_json_from_text(all_data['answer'])
        sorted_dict_desc = dict(sorted(knobs_score.items(), key=lambda item: item[1], reverse=True))
        logger.info(sorted_dict_desc)
        return list(sorted_dict_desc.keys())
    

    def load_data_prompt(self, knobs_num):
        system_prompt = textwrap.dedent(f"""
        Suppose you are an experienced DBA, and you are required to tune the knobs of {self.dbms.name}.
        "Use the following pieces of retrieved context to answer "
            "the question. "
            "\n\n"
            "{{context}}"
        """) 
        question = textwrap.dedent(f"""
        
        KNOB COLLECTION:{self.target_knobs}
        DBMS:{self.dbms.name}_{self.version}
        WORKLOAD: {self.benchmark}
        Hardware Information: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
        TASK:
        We have now used the historical data to generate some important knobs in KNOB COLLECTION that need to be further optimized in the subsequent fine-tuning process. However, the current list of knobs may also be missing important knobs for the current workload, or there may be cases where the one knob in the list depends on the other knobs which is not in KNOB COLLECTION. 
        For example, specifies the maximum delay in microseconds for the delay imposed when the innodb_max_purge_lag threshold is exceeded. The specified innodb_max_purge_lag_delay value is an upper limit on the delay period calculated by the innodb_max_purge_lag formula.
        Please generate up to five new knobs that are important in the current task but are not included in the knob collection, and they should be placed in order of importance from greatest to least
        Please check what each of these knobs in KNOB COLLECTION depends on, for example, if a modification of a certain range of knob values requires another knob to be enabled. 
        
        STEP:
        1. For each knob in KNOB COLLECTION, please make a judgment and find the other knobs which are interdependent. 
        2. Synthesize all the knobs suggested in the previous step, and assign each KNOB a rating from most important to least important to the Knob currently in the KNOB COLLECTION. 
        3. Consider whether there are any knobs that are not in the KNOB COLLECTION that are important to the current workload, and rate them by importance
        4. Combine the suggestion knobs from the two steps above to get the {knobs_num} most important knobs and rank them in order of importance
        Now let us think step by step and just return the the names of the suggested important knobs which are in the current dbms, but not in the knob collection list. Multiple knob names are separated by ','.
        NOTE:
        The final knob_names you suggested are different with the KNOB COLLECTION, but in {self.dbms.name}_{self.version}.
        
        Return ONLY the knob names as a comma-separated list.
        Example output format: max_worker_processes, parallel_tuple_cost, cpu_tuple_cost
        """
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        llm = ChatOpenAI(model=self.model_name)
       
        
        # prompt_llm = prompt | llm | self.extract_json
        all_data = llm.invoke({"input":question})

        # return the cost
        self.calc_token(all_data['input']+ ''.join(str(item) for item in all_data['context']), all_data['answer'])
        with open(self.cost_path, "a") as f:
            f.write("Select_RAG: \n")
            f.write("Total Tokens: "+ str(self.token))
            f.write("\n")
            f.write("Input Tokens: "+ str(self.input_token)+ ". Output Tokens: " + str(self.output_token))
            f.write("\n")
            f.write("Cost: "+ str(self.calc_money(self.input_token, self.output_token)))
            f.write("\n \n")

        final_knobs = self.target_knobs
        knob_list = [knob.strip("'").strip() for knob in all_data['answer'].split(",")]
        for k in knob_list:
            if k not in final_knobs:
                final_knobs.append(k)
        # for item_dict in all_data[0]["Knobs"]:
        #     if item_dict["interdependent_knobs"] == 'None':
        #         continue
        #     knob_list = [knob.strip() for knob in item_dict["interdependent_knobs"].split(",")]
        #     final_knobs.extend(knob_list)
        
        logger.info(f"Final Knobs: {final_knobs}")
        return final_knobs
    
    def load_data_self(self, knobs_num):
        system_prompt = textwrap.dedent(f"""
        Suppose you are an experienced DBA, and you are required to tune the knobs of {self.dbms.name}.
        "Use the following pieces of retrieved context to answer "
            "the question. "
            "\n\n"
            "{{context}}"
        """) 
        question = textwrap.dedent(f"""
        
        KNOB COLLECTION:{self.target_knobs}
        DBMS:{self.dbms.name}_{self.version}
        WORKLOAD: {self.benchmark}
        Hardware Information: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
        TASK:
        We have now used the historical data to generate some important knobs in KNOB COLLECTION that need to be further optimized in the subsequent fine-tuning process. However, the current list of knobs may also be missing important knobs for the current workload, or there may be cases where the one knob in the list depends on the other knobs which is not in KNOB COLLECTION. 
        For example, specifies the maximum delay in microseconds for the delay imposed when the innodb_max_purge_lag threshold is exceeded. The specified innodb_max_purge_lag_delay value is an upper limit on the delay period calculated by the innodb_max_purge_lag formula.
        Please generate up to five new knobs that are important in the current task but are not included in the knob collection, and they should be placed in order of importance from greatest to least
        Please check what each of these knobs in KNOB COLLECTION depends on, for example, if a modification of a certain range of knob values requires another knob to be enabled. 
        
        STEP:
        1. For each knob in KNOB COLLECTION, please make a judgment and find the other knobs which are interdependent. 
        2. Synthesize all the knobs suggested in the previous step, and assign each KNOB a rating from most important to least important to the Knob currently in the KNOB COLLECTION. 
        3. Consider whether there are any knobs that are not in the KNOB COLLECTION that are important to the current workload, and rate them by importance
        4. Combine the suggestion knobs from the two steps above to get the {knobs_num} most important knobs and rank them in order of importance
        Now let us think step by step and just return the the names of the suggested important knobs which are in the current dbms, but not in the knob collection list. Multiple knob names are separated by ','.
        NOTE:
        The final knob_names you suggested are different with the KNOB COLLECTION, but in {self.dbms.name}_{self.version}.
        Return ONLY the knob names as a comma-separated list.
        Example output format: max_worker_processes, parallel_tuple_cost, cpu_tuple_cost
        
        """
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        llm = ChatOpenAI(model=self.model_name)
        embeddings = OpenAIEmbeddings()

        # retriever
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        metadata_field_info = [
            AttributeInfo(
                name="knob_name",
                description="The name of knob which can be tuned in DBMS.",
                type="string"
            )
            # TODO add version
        ]
        document_content_description = "The information of knobs"
        retriever = SelfQueryRetriever.from_llm(
            llm,
            docsearch,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k": 5}
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        prompt_llm = create_retrieval_chain(retriever, question_answer_chain)
        
        # prompt_llm = prompt | llm | self.extract_json
        all_data = prompt_llm.invoke({"input":question})

        # return the cost
        self.calc_token(all_data['input']+ ''.join(str(item) for item in all_data['context']), all_data['answer'])
        with open(self.cost_path, "a") as f:
            f.write("Select_RAG: \n")
            f.write("Total Tokens: "+ str(self.token))
            f.write("\n")
            f.write("Input Tokens: "+ str(self.input_token)+ ". Output Tokens: " + str(self.output_token))
            f.write("\n")
            f.write("Cost: "+ str(self.calc_money(self.input_token, self.output_token)))
            f.write("\n \n")

        final_knobs = self.target_knobs
        knob_list = [knob.strip("'").strip() for knob in all_data['answer'].split(",")]
        for k in knob_list:
            if k not in final_knobs:
                final_knobs.append(k)
        # for item_dict in all_data[0]["Knobs"]:
        #     if item_dict["interdependent_knobs"] == 'None':
        #         continue
        #     knob_list = [knob.strip() for knob in item_dict["interdependent_knobs"].split(",")]
        #     final_knobs.extend(knob_list)
        
        logger.info(f"Final Knobs: {final_knobs}")
        return final_knobs

        
    def extract_json_from_text(self, text):
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, text)
        if match:
            try:
                json_data = json.loads(match.group())
                return json_data
            except json.JSONDecodeError:
                return None
        else:
            return None
    
    def load_data_graph(self, knobs_num):
        self.system_message = textwrap.dedent(f"""
        Suppose you are an experienced DBA, and you are required to tune the knobs of {self.dbms.name}.
        Execute the function 'query_graphRAG' to look for context.
        """) 
        self.search_type = "local"
        @tool
        def query_grahRAG(question: str) -> str:
            """Query the knob information and the main relationship from RAG search"""
            if self.search_type == 'local':
                logger.info('Local Search')
                ans = search_local(question)
            else:
                logger.info('Global Search')
                ans = search_global(question)
            return ans
        
        tools = [query_grahRAG]
        query = textwrap.dedent(f"""
        
        KNOB COLLECTION:{self.target_knobs}
        DBMS:{self.dbms.name}_{self.version}
        WORKLOAD: {self.worklod_info}
        DATA: {self.data_info}
        Hardware Information: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
        TASK:
        We have now used the historical data to generate some important knobs in KNOB COLLECTION that need to be further optimized in the subsequent fine-tuning process. However, the current list of knobs may also be missing important knobs for the current workload, or there may be cases where the one knob in the list depends on the other knobs which is not in KNOB COLLECTION. 
        For example, specifies the maximum delay in microseconds for the delay imposed when the innodb_max_purge_lag threshold is exceeded. The specified innodb_max_purge_lag_delay value is an upper limit on the delay period calculated by the innodb_max_purge_lag formula.
        Please generate up to five new dynamic knobs that are important in the current task but are not included in the knob collection, and they should be placed in order of importance from greatest to least
        Please check what each of these knobs in KNOB COLLECTION depends on, for example, if a modification of a certain range of knob values requires another knob to be enabled. 
        
        STEP:
        step1. For each knob in KNOB COLLECTION, please make a judgment and find the other knobs which are interdependent, according to the context from query_graphRAG. Execute the query as many times as there are as many knobs in the knob collection. 
        step2. Synthesize all the knobs suggested in the previous step, and assign each KNOB a rating from most important to least important to the Knob currently in the KNOB COLLECTION. 
        step3. Consider whether there are any knobs that are not in the KNOB COLLECTION that are important to the current workload, and rate them by importance
        step4. Combine the suggestion knobs from step1-3 and Exclude non-dynamic knobs. Then get at most {knobs_num} important knobs and rank them in order of importance
        Now let us think step by step and just return the the names of the suggested important knobs which are in the current dbms, but not in the knob collection list. Multiple knob names are separated by ','.
        NOTE:
        The final knob_names you suggested are different with the KNOB COLLECTION, but in {self.dbms.name}_{self.version}.
        Just return the dynamic knob names and split them with ', ', without other.
        Return ONLY the knob names as a comma-separated list.
        Example output format: max_worker_processes, parallel_tuple_cost, cpu_tuple_cost
        
        """
        )
        self.model = ChatOpenAI(model=self.model_name, temperature=0.1)
        app = create_react_agent(self.model, tools, messages_modifier=self.system_message)

        # messages = app.invoke({"messages": [("user", "what was the Suggest Agent suggestions?")]}, self.config,)
        # logger.info(messages["messages"][-1].content)

        messages = app.invoke({"messages": [("user", query)]})
        logger.info(messages)
        # {
        #     "input": query,
        #     "output": messages["messages"][-1].content,
        # }
        logger.info(messages["messages"][-1].content)

        self.token = messages["messages"][-1].usage_metadata['total_tokens']
        self.input_token = messages["messages"][-1].usage_metadata['input_tokens']
        self.output_token = messages["messages"][-1].usage_metadata['output_tokens']
        with open(self.cost_path, "a") as f:
            f.write("Select_RAG: \n")
            f.write("Total Tokens: "+ str(self.token))
            f.write("\n")
            f.write("Input Tokens: "+ str(self.input_token)+ ". Output Tokens: " + str(self.output_token))
            f.write("\n")
            f.write("Cost: "+ str(self.calc_money(self.input_token, self.output_token)))
            f.write("\n \n")

        final_knobs = self.target_knobs
        knob_list = [knob.strip("'").strip() for knob in messages["messages"][-1].content.split(",")]
        for k in knob_list:
            if k not in final_knobs:
                final_knobs.append(k)
        # for item_dict in all_data[0]["Knobs"]:
        #     if item_dict["interdependent_knobs"] == 'None':
        #         continue
        #     knob_list = [knob.strip() for knob in item_dict["interdependent_knobs"].split(",")]
        #     final_knobs.extend(knob_list)
        
        logger.info(f"Final Knobs: {final_knobs}")
        return final_knobs


            
