from pymilvus import connections, db, CollectionSchema


class VDBC:
    def __init__(self, alias='default', user='root', password='Milvus', host='localhost', port='19530',
                 db_name='default'):
        connections.connect(
            alias=alias,
            uri=f"{host}:{port}",
            token=f"{user}:{password}",
            db_name=db_name,
        )

    def create_db(self, db_name):
        '''创建一个数据库'''
        db.create_database(db_name)

    def create_collection(self, *args, **kwargs):
        '''创建一个集合'''
        book_id = CollectionSchema.FieldSchema(name="book_id", dtype="int64", is_primary=True, auto_id=True)
        book_name = CollectionSchema.FieldSchema(name="book_name", dtype="string", is_primary=False)
        word_count = CollectionSchema.FieldSchema(name="word_count", dtype="int64", is_primary=False)
        book_intro = CollectionSchema.FieldSchema(name="book_intro", dtype="string", is_primary=False)
        schema = CollectionSchema(
            fields=[book_id, book_name, word_count, book_intro],
            description="Test book search",
            enable_dynamic_field=True
        )
        collection_name = "book"

    def insert(self, vector, metadata):
        '''向向量数据库中插入一个向量'''
        self.collection_name.insert_one({'vector': vector, 'metadata': metadata})

    def disconnect(self, alias='default'):
        '''断开连接'''
        connections.disconnect(alias=alias)
