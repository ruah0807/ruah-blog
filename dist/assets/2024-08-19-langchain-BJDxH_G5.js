const n=`---
title: Supabase vector db생성 및 Langchain을 이용한 임베딩 문서 저장
---

## 프로젝트 배경 및 문제점

저는 최근 Supabase의 Edge Functions을 사용하여 서버리스 백엔드를 구축하는 프로젝트를 진행했습니다. 이 프로젝트는 TypeScript의 Deno 환경에서 구현할 수 있도록 설계되어 있었지만, 몇 가지 도전에 직면했습니다. 특히, Langchain을 사용해 RAG (Retrieval-Augmented Generation)를 구현해야 하는 상황에서 발생한 제약 사항들이 있었습니다.

#### 문제점

1. **Langchain의 Node.js 종속성:**
   Langchain의 여러 기능들이 Node.js 환경에서만 사용 가능하다는 사실이 큰 도전이었습니다. 특히, Deno 환경에서는 웹 문서를 로드하고 벡터화하는 작업을 수행할 수 없었고, 이는 프로젝트에 큰 제약이었습니다.

2. **웹 문서 로드와 벡터화 작업:**
   Edge Functions에서 웹 문서를 로드하려고 했지만, Langchain의 문서로드 관련 기능들이 “Node only”였기 때문에 사용할 수 없었습니다. 따라서, Python에서 웹 문서를 벡터화하여 Supabase 데이터베이스에 저장하는 방법으로 우회해야 했습니다.

3. **Python과 Supabase의 통합:**
   Python을 사용해 Supabase에 데이터를 저장하는 작업은 번거로웠지만, 필요한 데이터를 벡터화하고 저장하기 위해서는 이 방법이 최선이었습니다. 이후 저장된 데이터를 활용해 RAG를 구현할 계획입니다.

#### 해결 방법

- **Python을 통한 데이터 벡터화 및 저장:**
  Python의 Langchain 라이브러리를 활용해 웹 문서를 벡터화하고, 이 데이터를 Supabase에 저장했습니다. Supabase의 SupabaseVectorStore를 사용하여 벡터 데이터를 효율적으로 관리하기 위함입니다.
- **Edge Functions의 한계 극복:**
  Deno 환경의 제약 때문에 Python에서 처리한 데이터를 Supabase에 저장하고, 이 데이터를 Edge Functions에서 검색 및 활용할 수 있는 방식으로 설계했습니다. 이를 통해 Deno 환경에서 발생한 제약을 우회할 수 있었습니다.

이 글에서는 LangChain을 통해 문서를 벡터화하고, 이를 Supabase 데이터베이스에 저장하는 방법을 소개합니다.

이 모든것은 [Langchain의 공식문서](https://python.langchain.com/v0.2/docs/templates/rag-supabase/#setup-environment-variables)로부터 비롯하여 구현하였습니다.

![](https://velog.velcdn.com/images/looa0807/post/f50fe675-ed84-4634-855b-ef6bb32d2e9c/image.png)

---

## 테이블 생성

> 프로젝트를 진행하는 과정에서 Supabase의 웹페이지 대시보드에 있는 SQL Editor 환경을 통해 데이터베이스와 필요한 테이블을 생성했습니다. 이 환경에서 직접 SQL 쿼리를 작성하고 실행하여, pgvector 확장을 설치하고 문서 데이터를 저장할 수 있는 테이블을 설정했습니다. 이러한 방법으로 Langchain과의 통합을 준비했습니다.
> 이 환경은 [Supabase](https://supabase.com/)가 제공하는 웹 기반 SQL Editor로, 사용자에게 데이터베이스 설정과 관리를 직관적으로 처리할 수 있는 인터페이스를 제공합니다. 이를 통해 프로젝트에 필요한 데이터베이스 구조를 손쉽게 구성할 수 있었습니다.

### Supabase Dashboard 내의 SQL Editor

![](https://velog.velcdn.com/images/looa0807/post/296a110c-4f54-4299-8799-010f894a87d0/image.png)

### pgvector 확장 설치

Supabase에서 벡터 데이터를 저장하고 검색하기 위해서는 PostgreSQL의 확장 기능인 pgvector를 설치해야 합니다. 이는 벡터 데이터를 효율적으로 저장하고 처리할 수 있는 기능을 제공합니다.

\`\`\`sql
-- pgvector 확장 설치
create extension if not exists vector;
\`\`\`

### 문서 저장을 위한 테이블 생성

벡터화된 문서를 저장할 테이블을 생성합니다. 이 예시에서는 disease라는 테이블을 사용하여 문서 데이터를 저장합니다.

\`\`\`sql
create table
  disease (
    id uuid primary key,
    content text,
    metadata jsonb,
    embedding vector (384)
  );
\`\`\`

이 테이블은 다음과 같은 역할을 합니다:

- content: 문서의 실제 텍스트 데이터.
- metadata: 문서와 관련된 추가 정보(예: 출처, 카테고리).
- embedding: 텍스트 데이터를 벡터화한 결과.

### 문서 검색을 위한 함수 생성

생성된 임베딩 벡터를 기반으로 가장 관련성 높은 문서를 검색하기 위한 함수를 작성합니다.

\`\`\`typescript
create function match_disease_documents (
  query_embedding vector (384),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (disease.embedding <=> query_embedding) as similarity
  from disease
  where metadata @> filter
  order by disease.embedding <=> query_embedding;
end;
$$;
\`\`\`

이 함수는 LangChain을 통해 생성된 쿼리 임베딩과 데이터베이스에 저장된 문서들의 벡터 간 유사도를 계산하여 관련성 높은 문서를 검색합니다.

---

## LangChain과 Supabase

### 환경설정

이 코드는 Python을 사용하여 웹 페이지에서 문서를 로드하고, 이 문서들을 벡터화하여 Supabase 데이터베이스에 저장하는 작업을 수행합니다. 이 작업은 특히 AI 기반 검색 시스템을 구축하는 데 유용합니다. 이 코드를 더 잘 이해할 수 있도록 주요 단계들을 설명드리겠습니다.

\`\`\`py
from dotenv import load_dotenv
load_dotenv()
\`\`\`

dotenv 라이브러리를 사용하여 .env 파일에서 환경 변수를 로드합니다. 이 파일에는 SUPABASE_URL과 SUPABASE_SERVICE_KEY가 저장되어 있으며, Supabase에 접속하기 위한 URL과 API 키가 포함되어 있습니다.

### Supabase 클라이언트 생성

\`\`\`py
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)
\`\`\`

환경 변수에서 Supabase URL과 키를 Supabase 클라이언트를 생성합니다. 이 클라이언트를 통해 Supabase 데이터베이스와 상호작용할 수 있습니다.

### Hugging Face 임베딩 모델 설정

\`\`\`
embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')
\`\`\`

Hugging Face의 intfloat/multilingual-e5-small 모델을 사용하여 문서를 벡터화합니다. 이 모델은 문서를 384차원 벡터로 변환합니다.

### 웹 페이지에서 문서 로드

\`\`\`py
urls = [
    "https://example.com/doc1",  # 문서 URL
    "https://example.com/doc2",  # 문서 URL
]
loader = WebBaseLoader(urls)
documents = loader.load()
\`\`\`

WebBaseLoader를 사용하여 지정된 URL에서 문서를 로드합니다. 로드된 문서는 documents 변수에 저장됩니다.

### 텍스트 청크로 분할

\`\`\`py
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
\`\`\`

긴 문서를 검색에 적합한 크기로 청크(조각)로 분할합니다. 각 청크는 최대 1000자까지 포함하며, 청크 간의 중복은 200으로 설정하였습니다.

### 벡터화된 문서를 Supabase에 저장

\`\`\`py
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="테이블명"
)
\`\`\`

분할된 문서 청크를 벡터화하여 Supabase 데이터베이스의 disease 테이블에 저장합니다. 이 벡터화된 데이터는 이후 RAG검색에 활용됩니다.

> 이 코드의 주요 목적은 웹 페이지에서 문서를 자동으로 가져와, AI 기반의 모델을 사용해 이를 벡터로 변환한 후, Supabase에 저장하여 나중에 효율적으로 검색할 수 있도록 하는 것입니다.

### 저장된 데이터

![](https://velog.velcdn.com/images/looa0807/post/bc3caccb-ce24-4537-8c73-c6c7a3a18bb0/image.png)

---

벡터 검색은 단순한 키워드 매칭을 넘어, 의미적으로 유사한 문서를 찾을 수 있는 강력한 검색 방법입니다. 전통적인 검색 방법에서는 사용자가 입력한 키워드와 문서 내에 있는 단어들을 비교해 일치하는 문서를 찾아주지만, 벡터 검색은 문서의 의미적 유사성을 기반으로 관련 있는 문서들을 찾습니다. 이러한 검색 방식은 특히 AI 기반 애플리케이션에서 매우 유용하게 사용될 수 있으며, 사용자가 명확하게 표현하지 못한 의도까지도 파악해 더욱 정교한 검색 결과를 제공합니다.

벡터 검색을 통해 사용자는 단순한 키워드가 아닌, 문서의 전반적인 의미와 맥락을 반영한 검색 결과를 얻을 수 있습니다. 이는 자연어 처리, 추천 시스템, 검색 엔진 최적화 등 다양한 분야에서 중요한 역할을 할 수 있습니다. 이러한 이유로, AI와 빅데이터를 활용한 현대 애플리케이션에서는 벡터 검색이 점점 더 중요한 기술로 자리 잡고 있습니다.

따라서 이 코드는 AI와 데이터베이스 기술을 결합하여, 사용자에게 더 나은 검색 경험을 제공하기 위한 강력한 도구로 활용되기 위함입니다.
`;export{n as default};
