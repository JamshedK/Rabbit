from RAG.rag import RAG

from util.logger_config import logger
import textwrap

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
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder
# )
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
import lark
from langchain_community.document_loaders import JSONLoader
import json
import time
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import create_tool_calling_agent
from typing import List
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.output_parsers import StrOutputParser
# class Response(BaseModel):
#     """Final response to the question being asked"""
#     knobs: List[str] = Field(description="The target knobs asked in the question.")
#     suggested_values: List[List[str]] = Field(
#         description="Ever taget knob should have a list of suggested values for the question, and the order of list is the same as the order of the knobs list."
#         )
class Knob(BaseModel):
    knob_name: str = Field(..., description="The name of knob in Target Knobs")
    recommended_values: str = Field(...,description="The recommended values for the knob, and the values should be separate with ', '. These values should just be exact values with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept.")

class Answer(BaseModel):
    Knobs: List[Knob]

class InitRAG(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, candidate_knobs, init_configs_path):
        super().__init__(dbms, benchmark, vector_store_path)
        self.candidate_knobs = candidate_knobs
        self.init_configs_path = init_configs_path
    
    # def parse(output):
    #     # If no function was invoked, return to user
    #     if "function_call" not in output.additional_kwargs:
    #         return AgentFinish(return_values={"output": output.content}, log=output.content)

    #     # Parse out the function call
    #     function_call = output.additional_kwargs["function_call"]
    #     name = function_call["name"]
    #     inputs = json.loads(function_call["arguments"])

    #     # If the Response function was invoked, return to the user with the function inputs
    #     if name == "Response":
    #         return AgentFinish(return_values=inputs, log=str(function_call))
    #     # Otherwise, return an agent action
    #     else:
    #         return AgentActionMessageLog(
    #             tool=name, tool_input=inputs, log="", message_log=[output]
    #         )
    def check(self,data_dict, d):
        failed_knobs = []
        for knob, value in data_dict.items():
            if len(value) != d:
                failed_knobs.append(knob)
        return failed_knobs

    # now use it
    def load_data_chain(self, init_number):
        llm = ChatOpenAI(model=self.model_name,temperature=0)
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
            # add version
        ]
        document_content_description = "The information of knobs"
        retriever = SelfQueryRetriever.from_llm(
            llm,
            docsearch,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k": 6}
        )
        
        # rag_chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "You are a DBA. Please find and return the information about the all target knobs. Target knobs: {input} Information: This is the structured information for the above knobs:"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        
        chunk_candidate_knobs_list = [self.candidate_knobs[i:i+4] for i in range(0, len(self.candidate_knobs), 4)]

        all_data = {}
        i = 0
        question_system = textwrap.dedent(f"""
        Suppose you are an experienced DBA, and you are required to tune the knobs of {self.dbms.name}.
        Answer the user query. Output your answer as JSON that matches the given schema: ```json\n{{schema}}\n```. 
        Make sure to wrap the answer in ```json and ``` tags,
        """) 
        all_failed_knobs = []
        while i < len(chunk_candidate_knobs_list):
            chunk_candidate_knobs = chunk_candidate_knobs_list[i]
        # for chunk_candidate_knobs in chunk_candidate_knobs_list:
            question = """ 
            Target Knobs: %s
            Target Knobs Information: {information}
            Hardware Information:
            The machine running the postgres 14 dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
            -------------------
            Workload Information:
            The benchmark is %s.
            Data: %s
            -------------------
            Task：
            Please generate %d sets of knob configurations based on the characteristics of a current workload and the hardware information. These knob configurations will be used to initialize a Bayesian Optimization (BO) model. Each set should cover different adjustment strategiess to allow for finding the optimzal configurations during subsequent Bayesian Optimization process.

            Step:
            1. From the information of target knobs and your own knowledge, please generate setting suggestions for all knobs based on the current workload and hardware environment.
            2. Analyse the current workload and hardware information.
            3. Based on the suggestions, generate a set of configrations of all target knobs. These configurations should be within the acceptable range of the hardware environment and able to recognize the current task.
            4. Generate set 2 through set %d, ensuring each set exhibits noticeable differences to cover various tuning directions
            5. Modify your output based on hardware information. The value should not exceed the range that the hardware can bear and should not crash the database. 
            6. Arrange the all sets of recommended configuration values for target knobs and just return the answer in json format. The each knob_name comes from the target knob list, and each knob contains %d recommended values. 
            
            Let's step by step.
            """%(",".join(chunk_candidate_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info, init_number, init_number, init_number) 

            # prompt =  ChatPromptTemplate.from_template(question)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_system),
                    ("human", question),
                ]
            ).partial(schema=Answer.schema())
            overall_chain = {"information":rag_chain} | prompt | llm | self.extract_json

                # data = json.loads(result)
            result = overall_chain.invoke({"input": ",".join(chunk_candidate_knobs)})
            logger.info(result)
            data = {}
            try:
                items = result[0]["Knobs"]
                # logger.info(items)
                for item_dict in items:
                    knob_list = [knob.strip() for knob in item_dict["recommended_values"].split(",")]
                    # logger.info(knob_list)
                    data[item_dict["knob_name"]] = knob_list
                logger.info(data)
                
                # 需要检查每个旋钮生成建议值的数量，需要检查是否所有all knobs都有生成对应的建议值
                failed_knobs =  self.check(data,init_number)
                all_failed_knobs.extend(failed_knobs)
                i = i+1
                if len(failed_knobs) == 0:
                    all_data.update(data)
                else:
                    for knob, value in data.items():
                        if knob not in failed_knobs:
                            all_data[knob] = value

                    logger.info("Fail to generate for %s"%(",".join(failed_knobs)))
            except:
                logger.info("Fail to generate for %s, and try to generate again."%(",".join(chunk_candidate_knobs)))


        # for all failed knobs
        i=0
        chunk_failed_knobs_list = [all_failed_knobs[i:i+3] for i in range(0, len(all_failed_knobs), 3)]
        while i < len(chunk_failed_knobs_list):
            chunk_failed_knobs  = chunk_failed_knobs_list[i]
            question = """ 
            Target Knob: %s
            Target Knobs Information: {information}
            Hardware Information:
            The machine running the postgres 14 dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
            -------------------
            Workload Information:
            The benchmark is %s.
            -------------------
            Task：
            Please generate %d sets of knob configurations based on the characteristics of a current workload and the hardware information. These knob configurations will be used to initialize a Bayesian Optimization (BO) model. Each set should cover different adjustment strategiess to allow for finding the optimzal configurations during subsequent Bayesian Optimization process.

            Step:
            1. From the information of target knobs and your own knowledge, please generate setting suggestions for all knobs based on the current workload and hardware environment.
            2. Analyse the current workload and hardware information.
            3. Based on the suggestions, generate a set of configrations of all target knobs. These configurations should be within the acceptable range of the hardware environment and able to recognize the current task.
            4. Generate set 2 through set %d, ensuring each set exhibits noticeable differences to cover various tuning directions
            5. Modify your output based on hardware information. The value should not exceed the range that the hardware can bear and should not crash the database. 
            6. Arrange the all sets of recommended configuration values for target knobs and just return the answer in json format. The each knob_name comes from the target knob list, and each knob contains %d recommended values. 
            
            Let's step by step.
            """%(','.join(chunk_failed_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.benchmark, init_number, init_number, init_number) 

            # prompt =  ChatPromptTemplate.from_template(question)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_system),
                    ("human", question),
                ]
            ).partial(schema=Answer.schema())
            overall_chain = {"information":rag_chain} | prompt | llm | self.extract_json

                # data = json.loads(result)
            result = overall_chain.invoke({"input": ','.join(chunk_failed_knobs)})
            logger.info(result)
            data = {}
            try:
                items = result[0]["Knobs"]
                # logger.info(items)
                for item_dict in items:
                    knob_list = [knob.strip() for knob in item_dict["recommended_values"].split(",")]
                    # logger.info(knob_list)
                    data[item_dict["knob_name"]] = knob_list
                logger.info(data)
                
                # 需要检查每个旋钮生成建议值的数量，需要检查是否所有all knobs都有生成对应的建议值
                failed_knobs =  self.check(data,init_number)
                all_data.update(data)
                i = i+1
                logger.info("Fail to generate for %s, and try again"%(",".join(failed_knobs)) )
            except:
                logger.info("Fail to generate for %s, and try to generate again."%(",".join(chunk_failed_knobs)))
            
                

        with open(self.init_configs_path, "w") as json_file:
            json.dump(all_data,json_file, indent=2)

        # output the cost
        with open(self.cost_path, "a") as f:
            f.write("Init_RAG: \n")
            f.write("Total Tokens: "+ str(self.token))
            f.write("\n")
            f.write("Input Tokens: "+ str(self.input_token)+ ". Output Tokens: " + str(self.output_token))
            f.write("\n")
            f.write("Cost: "+ str(self.calc_money(self.input_token, self.output_token)))
            f.write("\n \n")

        

    
    
    def load_data(self, init_number):
        system_prompt = """
        Suppose you are an experienced DBA, and you are required to tune the knobs of %s.
        ---------------------------------------------------------
        """%(self.dbms.name)
        question_list = []

        chunk_candidate_knobs_list = [self.candidate_knobs[i:i+5] for i in range(0, len(self.candidate_knobs), 5)]

        for chunk_candidate_knobs in chunk_candidate_knobs_list:
            question = """
            Target knobs: %s
            Hardware Information:
            The machine running the postgres 14 dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
            -------------------
            Workload Information:
            The benchmark is %s.
            -------------------
            Task：
            Please generate %d sets of knob configurations based on the characteristics of a current workload and the hardware information. These knob configurations will be used to initialize a Bayesian Optimization (BO) model. Each set should cover different adjustment strategiess to allow for finding the optimzal configurations during subsequent Bayesian Optimization process.
            
            Step:
            1. Get the information of target knobs and generate the suggestions for setting these knobs.
            2. Analyse the current workload and hardware information.
            3. Generate a set of configrations of all target knobs.
            4. Generate set 2 through set %d, ensuring each set exhibits noticeable differences to cover various tuning directions
            5. Arrange the all sets of recommended configuration values for multiple target knobs and just return the answer in the following JSON RESULT TEMPLATE. The each knob_name comes from the target knob list, and each knob corresponds to a list containing %d recommended values. The number of key-value pairs of "knob_name: recommended_value_list" is determined by the number of target knobs.
            JSON RESULT TEMPLATE:
            {{
                "knob_name1": [], 
                "knob_name2": [], // These values should just be exact values that cannot contain %d and cannot be descriptive text, with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept. All values should be string type.
                "knob_name3": [],
                "knob_name4": [],
                "knob_name5": [],
                ....
            }},
            Let's step by step and just return the answer in JSON RESULT TEMPLATE.
            """%(",".join(chunk_candidate_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.benchmark, init_number, init_number, init_number, init_number)        
            question_list.append(question)

        all_data_list = self.get_answer(system_prompt, question_list)
        all_data = {}
        for data in all_data_list:
            all_data.update(data)
        with open(self.init_configs_path, "w") as json_file:
            json.dump(all_data,json_file, indent=2)

    def init_agent(self, init_number):
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 5})
        retriever_tool = create_retriever_tool(
            retriever,
            "knob_search",
            "Search for information about the specific knob. For any questions about knobs, you must use this tool!"
        )
        llm = ChatOpenAI(model=self.model_name, temperature=0)
        # llm_with_tools = llm.bind_functions([retriever_tool, Response])
        llm_with_tools = llm.bind_functions([retriever_tool])

        # Get the prompt to use - you can modify this!
        from langchain import hub
        from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are an experienced DBA"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                # Format agent scratchpad from intermediate steps
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
        from langchain.agents import AgentExecutor
        agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

        chunk_candidate_knobs_list = [self.candidate_knobs[i:i+5] for i in range(0, len(self.candidate_knobs), 5)]
        for chunk_candidate_knobs in chunk_candidate_knobs_list:
            question = """
            Suppose you are an experienced DBA, and you are required to tune a knob of %s.
            ---------------------------------------------------------
            Target knobs: %s
            Hardware Information:
            The machine running the postgres 14 dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
            -------------------
            Workload Information:
            The benchmark is %s.
            -------------------
            Task：
            Please generate %d sets of knob configurations based on the characteristics of a current workload and the hardware information. These knob configurations will be used to initialize a Bayesian Optimization (BO) model. Each set should cover different adjustment strategiess to allow for finding the optimzal configurations during subsequent Bayesian Optimization process.
            
            Step:
            1. Get the information of target knobs.
            2. Analyse the current workload and hardware information.
            3. Generate a set of configrations of all target knobs.
            4. Generate set 2 through set %d, ensuring each set exhibits noticeable differences to cover various tuning directions
            5. Arrange the all sets of recommended configuration values for multiple target knobs and just return the answer in the following JSON RESULT TEMPLATE. The each knob_name comes from the target knob list, and each knob corresponds to a list containing %d recommended values. The number of key-value pairs of "knob_name: recommended_value_list" is determined by the number of target knobs.
            JSON RESULT TEMPLATE:
            {{
            "knob_name1": [], 
            "knob_name2": [], // These values should just be exact values that cannot contain %d and cannot be descriptive text, with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept. All values should be string type.
            "knob_name3": [],
            "knob_name4": [],
            "knob_name5": [],
            }},
            Let's step by step and just return the answer in JSON RESULT TEMPLATE.
            """%(self.dbms.name, ",".join(chunk_candidate_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.benchmark, init_number, init_number, init_number, init_number)
            ans = agent_executor.invoke({"input": question},
                                        return_only_outputs=True)
            # ans = list(agent_executor.stream({"input":question}))

            logger.info(ans)
            break



        