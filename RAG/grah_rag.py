import os
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
import pandas as pd
from IPython.display import display
import tiktoken
import asyncio
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
import time
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from yfiles_jupyter_graphs import GraphWidget
import matplotlib.pyplot as plt
def convert_entities_to_dicts(df):
    """Convert the entities dataframe to a list of dicts for yfiles-jupyter-graphs."""
    nodes_dict = {}
    for _, row in df.iterrows():
        # Create a dictionary for each row and collect unique nodes
        node_id = row["title"]
        if node_id not in nodes_dict:
            nodes_dict[node_id] = {
                "id": node_id,
                "properties": row.to_dict(),
            }
    return list(nodes_dict.values())


# converts the relationships dataframe to a list of dicts for yfiles-jupyter-graphs
def convert_relationships_to_dicts(df):
    """Convert the relationships dataframe to a list of dicts for yfiles-jupyter-graphs."""
    relationships = []
    for _, row in df.iterrows():
        # Create a dictionary for each row
        relationships.append({
            "start": row["source"],
            "end": row["target"],
            "properties": row.to_dict(),
        })
    return relationships
    

def search_local(q):

    api_key = os.environ["OPENAI_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    api_base = os.environ["PROXY"]
    embedding_model = "text-embedding-3-small"
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_base = api_base,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=50,
    )
    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=api_base,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=50,
    )

    # parquet files generated from indexing pipeline
    INPUT_DIR = "./knowledge/postgres/graph/output/artifacts"
    LANCEDB_URI = "./knowledge/postgres/graph/output/lancedb"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    # COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2
    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

# load description embeddings to an in-memory lancedb vectorstore
# to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    for attempt in range(5):
        try:
            entity_description_embeddings = store_entity_semantic_embeddings(
                entities=entities, vectorstore=description_embedding_store
            )
            break
        except OSError as e:
            continue
    else:
        raise Exception("Max retries reached. Dataset write operation failed due to persistent commit conflicts.")


    # print(f"Entity count: {len(entity_df)}")
    # print(entity_df.head())

    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    # print(f"Relationship count: {len(relationship_df)}")
    # print(relationship_df.head())

    # NOTE: covariates are turned off by default, because they generally need prompt tuning to be valuable
    # Please see the GRAPHRAG_CLAIM_* settings
    # covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")

    # claims = read_indexer_covariates(covariate_df)

    # print(f"Claim records: {len(claims)}")
    # covariates = {"claims": claims}

    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    # print(f"Report records: {len(report_df)}")
    # print(report_df.head())

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    # print(f"Text unit records: {len(text_unit_df)}")
    # print(text_unit_df.head())

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        # if you did not run covariates during indexing, set this to None
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 1,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 3,
        "top_k_relationships": 3,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )


    async def search(query):
        result = await search_engine.asearch(
        query
    )
        print(result.response)
        print(result.context_data["entities"].head())
        print(result.context_data["relationships"].head())
        # print(result.context_data["reports"].head())
        # print(result.context_data["sources"].head())
        # if "claims" in result.context_data:
        #     print(result.context_data["claims"].head())
        return result.response
    ans = asyncio.run(search(q))
    return ans

def search_global(q):

    api_key = os.environ["OPENAI_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    api_base = os.environ["PROXY"]

    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Global
    # parquet files generated from indexing pipeline
    INPUT_DIR = "./knowledge/postgres/graph/output/artifacts"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2

    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    print(report_df.head())

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )
    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    async def search(query):
        result = await search_engine.asearch(
        query
    )
        print(result.response)
        result.context_data["reports"]
        # inspect number of LLM calls and tokens
        print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")

    ans = asyncio.run(search(q))
    return ans

# if __name__ == '__main__':
#     q = "What is 'innodb_buffer_pool_size', and what are his main relationships?"
#     q = "Which knobs are important for the workload of tpcc?"
#     search_local(q)




