import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from app.services.messages import NO_VERSES_FOUND

# 尝试不同的 PromptTemplate 导入路径
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        try:
            from langchain.prompts.prompt import PromptTemplate
        except ImportError:
            # 如果都失败，使用字符串模板
            PromptTemplate = None

# 尝试不同的 load_qa_chain 导入路径
try:
    from langchain.chains import load_qa_chain
except ImportError:
    try:
        from langchain.chains.question_answering import load_qa_chain
    except ImportError:
        load_qa_chain = None

# 尝试不同的 RetrievalQA 导入路径
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
except ImportError:
    try:
        from langchain.chains.retrieval_qa import RetrievalQA
    except ImportError:
        try:
            from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
            RetrievalQA = None  # 如果找不到 RetrievalQA，使用备选方案
        except ImportError:
            RetrievalQA = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bible-verses")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")


class RAGService:
    """使用 LangChain 实现的 RAG 服务"""
    
    def __init__(self):
        # 初始化 embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        
        # 初始化 Pinecone Vector Store
        self.vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 创建 retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.5}
        )
        
        # 定义 prompt 模板字符串
        qa_template = """You are a careful Bible study assistant. 
You can ONLY answer based on the provided Bible verses. 
If the context is insufficient, say you are not sure and encourage the user to read the Bible directly.

Context:
{context}

Question: {question}

Answer:"""
        
        qa_template_alt = """You are a careful Bible study assistant. 
You can ONLY answer based on the provided Bible verses. 
If the context is insufficient, say you are not sure and encourage the user to read the Bible directly.

Use the following pieces of context to answer the question. If you don't know the answer, say you are not sure and encourage the user to read the Bible directly.

{context}

Question: {question}

Answer:"""
        
        # 创建自定义 prompt（用于 RetrievalQA）
        if PromptTemplate is not None:
            self.qa_prompt = PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            )
            self.qa_prompt_alt = PromptTemplate(
                template=qa_template_alt,
                input_variables=["context", "question"]
            )
        else:
            # 如果 PromptTemplate 不可用，存储模板字符串
            self.qa_prompt = qa_template
            self.qa_prompt_alt = qa_template_alt
        
        # 创建 RetrievalQA chain
        if RetrievalQA is not None and isinstance(self.qa_prompt, PromptTemplate):
            # 使用 RetrievalQA（如果可用）
            try:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": self.qa_prompt},
                    return_source_documents=True,
                    verbose=False
                )
            except Exception:
                self.qa_chain = None
        else:
            self.qa_chain = None
        
        # 如果 RetrievalQA 不可用，使用 load_qa_chain 或手动实现
        if self.qa_chain is None:
            if load_qa_chain is not None and isinstance(self.qa_prompt_alt, PromptTemplate):
                try:
                    self.qa_prompt_chain = load_qa_chain(
                        llm=self.llm,
                        chain_type="stuff",
                        prompt=self.qa_prompt_alt,
                        verbose=False
                    )
                except Exception:
                    self.qa_prompt_chain = None
            else:
                self.qa_prompt_chain = None
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        查询问题并返回答案和相关经文
        
        Returns:
            {
                "answer": str,
                "verses": List[Dict]  # 包含 book, chapter, verse, text
            }
        """
        if self.qa_chain is not None:
            # 使用 RetrievalQA chain
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
        elif self.qa_prompt_chain is not None:
            # 使用 load_qa_chain
            # 使用 invoke 方法（新版本 LangChain）或 get_relevant_documents（旧版本）
            try:
                docs = self.retriever.invoke(question)
            except (AttributeError, TypeError):
                # 兼容旧版本
                docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return {
                    "answer": NO_VERSES_FOUND,
                    "verses": []
                }
            
            result = self.qa_prompt_chain.invoke({
                "input_documents": docs,
                "question": question
            })
            answer = result["output_text"]
            source_docs = docs
        else:
            # 完全手动实现：使用 retriever + LLM 直接调用
            # 使用 invoke 方法（新版本 LangChain）或 get_relevant_documents（旧版本）
            try:
                docs = self.retriever.invoke(question)
            except (AttributeError, TypeError):
                # 兼容旧版本
                docs = self.retriever.get_relevant_documents(question)
            if not docs:
                return {
                    "answer": NO_VERSES_FOUND,
                    "verses": []
                }
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 格式化 prompt（使用字符串模板）
            if isinstance(self.qa_prompt_alt, str):
                prompt_text = self.qa_prompt_alt.format(context=context, question=question)
            else:
                # 如果是 PromptTemplate 对象，使用 format_prompt
                prompt_text = self.qa_prompt_alt.format(context=context, question=question)
            
            # 使用 LLM 生成答案
            # 尝试导入消息类型，如果失败则使用字典格式
            try:
                from langchain.schema import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content="You are a careful Bible study assistant."),
                    HumanMessage(content=prompt_text)
                ]
            except ImportError:
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [
                        SystemMessage(content="You are a careful Bible study assistant."),
                        HumanMessage(content=prompt_text)
                    ]
                except ImportError:
                    # 使用字典格式（ChatOpenAI 也支持）
                    messages = [
                        {"role": "system", "content": "You are a careful Bible study assistant."},
                        {"role": "user", "content": prompt_text}
                    ]
            
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            source_docs = docs
        
        # 提取源文档（经文）
        verses = []
        seen_ids = set()  # 跟踪已使用的 id，避免重复
        
        for doc in source_docs:
            metadata = doc.metadata
            book = metadata.get("book", "")
            chapter = int(metadata.get("chapter", 0))
            verse = int(metadata.get("verse", 0))
            
            # 优先使用 metadata 中的 id，如果没有则生成一个
            verse_id = metadata.get("id", "")
            if not verse_id and book and chapter and verse:
                verse_id = f"{book}.{chapter}.{verse}"
            
            # 如果 id 仍然为空或已存在，添加后缀确保唯一性
            if not verse_id:
                verse_id = f"verse-{len(verses)}"
            elif verse_id in seen_ids:
                verse_id = f"{verse_id}-{len(verses)}"
            
            seen_ids.add(verse_id)
            
            verses.append({
                "id": verse_id,
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "text": metadata.get("text", doc.page_content)
            })
        
        return {
            "answer": answer,
            "verses": verses
        }
    
    def get_retriever(self):
        """返回 retriever 用于自定义查询"""
        return self.retriever


# 全局实例
_rag_service = None

def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

