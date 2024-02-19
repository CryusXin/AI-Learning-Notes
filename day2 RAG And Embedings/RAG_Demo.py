import os

from openai.lib.azure import AzureOpenAI
from pymilvus import connections, db, CollectionSchema, DataType, FieldSchema, utility, Collection
from sentence_transformers import CrossEncoder
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from processing_pdf import PDFProcessor


class VDBC:
    def __init__(self, alias='default', user='root', password='Milvus', host='192.168.100.251', port='19530',
                 db_name='default'):
        connections.connect(
            alias=alias,
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name,
        )

    def create_db(self, db_name):
        '''创建一个数据库'''
        db.create_database(db_name)

    def disconnect(self, alias='default'):
        '''断开连接'''
        connections.disconnect(alias=alias)

    def create_collection(self, collection_name):
        '''创建一个集合'''
        text_id = FieldSchema(
            name="text_id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        text_content = FieldSchema(
            name="text_content",
            dtype=DataType.VARCHAR,
            max_length=2000,
            default_value="Unknown"
        )
        text_embeddings = FieldSchema(
            name="text_embeddings",
            dtype=DataType.FLOAT_VECTOR,
            dim=384
        )
        schema = CollectionSchema(
            fields=[text_id, text_content, text_embeddings],
            description="Test pdf search",
            enable_dynamic_field=True
        )
        Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=2,
        )

    def insert(self, collection_name, data):
        '''插入数据'''
        Collection(collection_name).insert(data)

    def create_index(self, collection_name):
        '''创建索引'''
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection = Collection("pdf_text")
        collection.create_index(
            field_name="text_embeddings",
            index_params=index_params
        )
        utility.index_building_progress("pdf_text")

    def search(self, query, limit=2):
        '''检索向量数据库'''
        search_params = {
            "metric_type": "COSINE",  # 指定相似度度量类型 L2, IP, COSINE
            "offset": 0,  # 返回结果的偏移量
            "ignore_growing": False,  # 是否忽略未完成的索引
            "params": {"nlist": 1024}  # 搜索参数
        }
        db.using_database('wangxin')  # 使用数据库
        collection = Collection("pdf_text")  # 加载集合
        collection.load()  # 加载集合
        results = collection.search(
            data=get_embedding([query]),  # 查询向量
            anns_field="text_embeddings",  # 查询字段
            param=search_params,  # 搜索参数
            limit=limit,  # 返回结果的数量
            expr=None,  # 查询表达式
            output_fields=['text_content'],  # 输出字段
            consistency_level="Strong"  # 一致性级别
        )

        # 检索结果排序
        res = sort_results(results, user_query)
        return [i[1] for i in res]


# 构建一个 RAG_BOT 类
class RAG_BOT:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt_template = """
            你是一个问答机器人。
            你的任务是根据下述给定的已知信息回答用户问题。
            确保你的回复完全依据下述已知信息。不要编造答案。
            如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

            已知信息:
            __INFO__

            用户问：
            __QUERY__

            请用中文回答用户问题。
            """
        prompt = build_prompt(
            prompt_template, info=search_results, query=user_query)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response


# 获取文本向量
def get_embedding(sentences):
    '''获取文本的向量表示'''
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Print the embeddings
    return embeddings


# 检索后排序
def sort_results(results, user_query):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    scores = model.predict([(user_query, doc.entity.get('text_content')) for doc in results[0]])
    # 按得分排序
    sorted_list = sorted(zip(scores, [doc.entity.get('text_content') for doc in results[0]]), key=lambda x: x[0],
                         reverse=True)
    return sorted_list
    # print(sorted_list)
    # for score, doc in sorted_list:
    #     print(f"{score}\t{doc}\n")


# 构建 Prompt
def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt


# 调用 Azure openai 接口
def get_completion(prompt):
    '''封装 Azure openai 接口'''
    api_base = 'https://micker-gpt-4-tubro.openai.azure.com/'  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    api_key = "2640762c94384b6a98436a8048d84670"
    deployment_name = 'gpt-4-turbo'
    api_version = '2023-07-01-preview'  # this might change in the future

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    vdbc = VDBC()
    # vdbc.create_db('wangxin') # 创建数据库
    # print(db.list_database())
    db.using_database('wangxin')  # 使用数据库
    # vdbc.create_collection('pdf_text')  # 创建集合
    # utility.drop_collection('pdf_text') # 删除集合
    # print(utility.list_collections())

    # pdfp = PDFProcessor()  # 创建一个 PDFProcessor 实例
    # paragraphs = pdfp.extract_text_from_pdf("llama2.pdf", min_line_length=10)
    # chunks = pdfp.split_text(paragraphs, 300, 100)
    # embeddings = get_embedding(chunks)
    # if len(chunks) > 0:
    #     ids = [i for i in range(len(chunks))]
    #     print(f'开始写入向量数据')
    #     vdbc.insert('pdf_text', [ids, chunks, embeddings])

    # vdbc.create_index('pdf_text')  # 创建索引
    # user_query = "Llama 2有多少参数"

    # 测试RAG_BOT
    rag_bot = RAG_BOT(vdbc, get_completion)
    # user_query = "Llama 2有多少参数"
    user_query = "llama 2有对话版吗？"
    print(f"用户：{user_query}")
    response = rag_bot.chat(user_query)
    print(f"RAG_BOT：{response}")