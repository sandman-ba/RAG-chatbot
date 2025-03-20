from typing import List, Tuple
from llama_index.core import PromptTemplate
from transformers import AutoModelForSequenceClassification

ranker_model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)

ranker_model.to("cuda")
ranker_model.eval()

table_declarations = [# TODO: Change to fit our data
    "CREATE TABLE platform (\n\tid INTEGER PRIMARY KEY,\n\tplatform_name TEXT DEFAULT NULL\n);",
    "CREATE TABLE genre (\n\tid INTEGER PRIMARY KEY,\n\tgenre_name TEXT DEFAULT NULL\n);",
    "CREATE TABLE publisher (\n\tid INTEGER PRIMARY KEY,\n\tpublisher_name TEXT DEFAULT NULL\n);",
    "CREATE TABLE region (\n\tid INTEGER PRIMARY KEY,\n\tregion_name TEXT DEFAULT NULL\n);",
    "CREATE TABLE game (\n\tid INTEGER PRIMARY KEY,\n\tgenre_id INTEGER,\n\tgame_name TEXT DEFAULT NULL,\n\tCONSTRAINT fk_gm_gen FOREIGN KEY (genre_id) REFERENCES genre(id)\n);",
    "CREATE TABLE game_publisher (\n\tid INTEGER PRIMARY KEY,\n\tgame_id INTEGER DEFAULT NULL,\n\tpublisher_id INTEGER DEFAULT NULL,\n\tCONSTRAINT fk_gpu_gam FOREIGN KEY (game_id) REFERENCES game(id),\n\tCONSTRAINT fk_gpu_pub FOREIGN KEY (publisher_id) REFERENCES publisher(id)\n);",
    "CREATE TABLE game_platform (\n\tid INTEGER PRIMARY KEY,\n\tgame_publisher_id INTEGER DEFAULT NULL,\n\tplatform_id INTEGER DEFAULT NULL,\n\trelease_year INTEGER DEFAULT NULL,\n\tCONSTRAINT fk_gpl_gp FOREIGN KEY (game_publisher_id) REFERENCES game_publisher(id),\n\tCONSTRAINT fk_gpl_pla FOREIGN KEY (platform_id) REFERENCES platform(id)\n);",
    "CREATE TABLE region_sales (\n\tregion_id INTEGER DEFAULT NULL,\n\tgame_platform_id INTEGER DEFAULT NULL,\n\tnum_sales REAL,\n   CONSTRAINT fk_rs_gp FOREIGN KEY (game_platform_id) REFERENCES game_platform(id),\n\tCONSTRAINT fk_rs_reg FOREIGN KEY (region_id) REFERENCES region(id)\n);",
]

make_sql_prompt_tmpl_text = """
Generate a SQL query to answer the following question from the user:
\"{query_str}\"

The SQL query should use only tables with the following SQL definitions:

Table 1:
{table_1}

Table 2:
{table_2}

Table 3:
{table_3}

Make sure you ONLY output an SQL query and no explanation.
"""
make_sql_prompt_tmpl = PromptTemplate(make_sql_prompt_tmpl_text)

def rank_tables(query: str, table_specs: List[str], top_n: int = 0) -> List[Tuple[float, str]]:
    """
    Get sorted pairs of scores and table specifications, then return the top N,
    or all if top_n is 0 or default.
    """
    pairs = [[query, table_spec] for table_spec in table_specs]
    scores = ranker_model.compute_score(pairs)
    scored_tables = [(score, table_spec) for score, table_spec in zip(scores, table_specs)]
    scored_tables.sort(key=lambda x: x[0], reverse=True)
    if top_n and top_n < len(scored_tables):
        return scored_tables[0:top_n]
    return scored_tables

def answer_query(user_query: str) -> str:
    try:
        ranked_tables = rank_tables(user_query, table_declarations, top_n=3)
    except Exception as e:
        print(f"Ranking failed.\nUser query:\n{user_query}\n\n")
        raise(e)

    make_sql_prompt = make_sql_prompt_tmpl.format(
        query_str=user_query, table_1=ranked_tables[0][1], table_2=ranked_tables[1][1], table_3=ranked_tables[2][1]
    )

    try:
        response = bloom_llm.complete(make_sql_prompt)
    except Exception as e:
        print(f"SQL query generation failed\nPrompt:\n{make_sql_prompt}\n\n")
        raise(e)

    sql_query = str(response).replace("\\", "")

    try:
        sql_response = sqlite3.connect("database.db").cursor().execute(sql_query).fetchall()
    except Exception as e:
        print(f"SQL querying failed.\nQuery:\n{sql_query}\n\n")
        raise(e)

    rag_prompt = rag_prompt_tmpl.format(query_str=user_query, json_table=json.dumps(sql_response), sql_query=sql_query)

    try:
        rag_response = bloom_llm.complete(rag_prompt)
        return str(rag_response)
    except Exception as e:
        print(f"Answer generation failed.\nPrompt:\n{rag_prompt}\n\n")
        raise(e)
