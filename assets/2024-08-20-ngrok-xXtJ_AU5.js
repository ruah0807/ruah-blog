const n=`---
title: FastAPI와 ngrok으로 Self-Query RAG 구현
---

## Edge Functions에서의 한계 극복과 FastAPI로의 전환

이 글에서는 FastAPI를 활용해 Self-Query RAG(Retrieval-Augmented Generation)를 구현한 경험을 공유합니다. 처음에는 Deno 기반의 Edge Functions에서 LangChain을 사용해 Self-Query RAG를 구현하려 했으나, 여러 문제에 직면하게 되었습니다. 이를 해결하기 위해 FastAPI로 전환하게 되었으며, ngrok을 사용해 로컬에서 개발한 FastAPI 서버를 Edge Functions와 통합하는 방법을 소개합니다.

### Edge Functions에서의 도전과 한계

초기 목표는 Deno의 Edge Functions에서 LangChain을 사용해 Self-Query RAG를 구현하는 것이었습니다. 그러나, 아래와 같은 문제가 발생하면서 계획을 수정해야 했습니다.

**1.1 NPM 라이브러리와 Deno의 호환성 문제**
Supabase의 공식문서"[Edge Functions: Node and native npm compatibility](https://supabase.com/blog/edge-functions-node-npm)"를 통해 npm을 통해 필요한 라이브러리들을 설치하고, 이를 import_map.json에 추가하여 Edge Functions에서 사용하려 했습니다. 그러나 다음과 같은 문제가 발생했습니다:

- 엔드포인트 미작동: Edge Functions에서 npm 라이브러리들이 정상적으로 작동하지 않아 404 에러만 발생하고 아무 log도 볼 수 없는 문제가 발생했습니다.

**1.2 VectorStore와 Self-QueryRetriever의 버전 호환성 문제**
이후 esm.sh와 같은 CDN을 통해 Deno 환경에 맞는 버전의 라이브러리를 불러와 사용하는 방법을 선택하였고, 여전히 VectorStore와 Self-QueryRetriever 간의 버전 충돌 문제를 해결할 수 없었습니다.

결론: 이러한 문제들로 인해 Edge Functions 환경에서 Self-Query RAG를 구현하는 것은 비효율적이라는 결론에 도달했습니다. 이에 따라 FastAPI로 전환하여 문제를 해결하기로 결정했습니다.

### FastAPI로의 전환

**1.1 FastAPI를 사용한 Self-Query RAG 구현**

FastAPI는 Python 기반의 웹 프레임워크로, REST API를 빠르고 쉽게 구축할 수 있게 해줍니다. LangChain의 Self-Query RAG를 구현하는 데에도 매우 적합한 도구였습니다.

FastAPI에서 Self-Query RAG를 구현하기 위해 다음과 같은 과정을 거쳤습니다:

1. LangChain 설치 및 초기화: Python 환경에서 LangChain을 설치하고, 필요한 LLM과 임베딩 모델을 초기화했습니다.
2. FastAPI 서버 구축: REST API를 통해 설문 데이터와 레시피 데이터를 받아, 이를 기반으로 맞춤형 가이드를 생성하는 API 엔드포인트를 구축했습니다.
3. Self-QueryRetriever 설정: LangChain의 Self-QueryRetriever를 사용해, 사용자의 질병 상태와 라이프스타일에 맞는 건강 정보를 검색하고, 이를 기반으로 가이드를 생성했습니다.

---

## 구현과정

### 1. FastAPI 환경설정

먼저 몇가지 환경설정과 라이브러리를 설치합시다

\`\`\`python
pip install fastapi uvicorn
pip install langchain pinecone-client openai
\`\`\`

### 2. Langchain과 Self-Query RAG 구현

이미 이전에 임베딩을통해 관련 웹 문서들을 백터화해 Pinecone에 저장해둔 상태이므로 검색부터 진행하였습니다.
LangChain을 통해 Self-Query RAG를 구현하기 위해서는 임베딩 모델과 LLM(Language Learning Model)을 설정하고, 이를 통해 적절한 검색과 텍스트 생성을 진행해야 합니다.

\`\`\`python
from fastapi import FastAPI, HTTPException, Request, Depends
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from sqlalchemy.orm import Session

app = FastAPI()

# 임베딩 모델 설정
embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')

# 벡터 스토어 설정
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name="disease"
)

# Self-QueryRetriever 설정
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(api_key='your-api-key', model='gpt-4o-mini'),
    vectorstore=vector_store,
    document_contents="건강관련 식사요법",
    metadata_field_info=[
        {"name": "category", "description": "건강정보 카테고리", "type": "string"},
        {"name": "source", "description": "건강정보 소스", "type": "string"},
    ]
)
\`\`\`

#### 2-1. 임베딩모델설정

임베딩 모델은 입력된 텍스트를 벡터 형태로 변환하여, 문서 검색 및 RAG에서 유사도를 계산할 수 있도록 도와줍니다. 이 프로젝트에서는 Hugging Face의 intfloat/multilingual-e5-small 모델을 사용했습니다. 이 모델은 다양한 언어를 지원하며, 작은 크기에도 불구하고 좋은 성능을 보입니다.

\`\`\`
    # 임베딩 모델 설정
    embeddings = HuggingFaceTransformersEmbeddings(model_name='intfloat/multilingual-e5-small')
\`\`\`

HuggingFaceTransformersEmbeddings 클래스를 사용하여 임베딩 객체를 생성합니다. 모델 이름으로 ‘intfloat/multilingual-e5-small’을 지정하여, 해당 모델을 로드하고 사용할 수 있도록 합니다. 이 임베딩 객체는 이후에 벡터 스토어와 통합되어 문서 검색에서 사용됩니다.

#### 2-2. 검색 로직 설정 (Self-Query Retriever)

Self-Query RAG는 사용자가 입력한 질의와 일치하는 문서를 벡터 검색을 통해 찾아내고, 이를 바탕으로 사용자가 원하는 정보를 생성하는 방식입니다. 여기서 중요한 부분은 Self-Query Retriever를 설정하는 것입니다.

\`\`\`python
    # Self-QueryRetriever 설정
    retriever = SelfQueryRetriever.from_llm(
        llm=ChatOpenAI(api_key='your-api-key', model='gpt-4o-mini'),
        vectorstore=vector_store,
        document_contents="건강관련 식사요법",
        metadata_field_info=[
            {"name": "category", "description": "건강정보 카테고리", "type": "string"},
            {"name": "source", "description": "건강정보 소스", "type": "string"},
        ]
    )
\`\`\`

- LLM 설정: OpenAI의 GPT-4 모델을 사용하여 입력된 질의를 분석하고, 적절한 검색 쿼리를 생성합니다.
- Vector Store와 통합: 생성된 임베딩 객체를 이용해 벡터 스토어에 저장된 문서를 검색합니다. 이 과정에서 사용자는 자연어로 질문을 입력할 수 있으며, 이 질문은 검색 가능한 쿼리로 변환됩니다.
- 메타데이터 설정: 문서의 메타데이터 필드를 지정하여, 검색 시 특정 카테고리나 출처를 필터링할 수 있습니다. 예를 들어, “건강정보 카테고리”와 “건강정보 소스”를 필터로 설정하여 더 정확한 검색 결과를 얻을 수 있습니다.

### 3. 데이터베이스와 통합

FastAPI를 사용하여 사용자로부터 설문 데이터와 레시피 데이터를 입력받아, 이를 바탕으로 맞춤형 가이드를 생성하는 기능을 구현했습니다. 데이터베이스와의 통합은 아래와 같이 이루어집니다.

\`\`\`python
@app.post("/generate-guide/")
async def generate_custom_guide(surveyId: int, recipeId: int, db: Session = Depends(get_db)):
    # 설문 데이터 가져오기
    survey_data = db.execute(text("SELECT health, eating_habits, excluded_ingredients FROM survey WHERE id = :id"),
                             {"id": surveyId}).fetchone()

    # 레시피 데이터 가져오기
    recipe_data = db.execute(text("SELECT title, ingredients, instructions FROM recipe WHERE id = :id"),
                             {"id": recipeId}).fetchone()

    # 문서 검색 및 가이드 생성
    query = f"{survey_data['health']} 식사 가이드"
    relevant_docs = retriever.get_relevant_documents(query)
    relevant_info = "\\n".join([doc.page_content for doc in relevant_docs])

    # 가이드 생성
    guide = create_prompt(survey_data, recipe_data, relevant_info)
    return {"guide": guide}
\`\`\`

#### 3-1. 설문 데이터와 레시피 데이터 가져오기

FastAPI에서 설문 데이터와 레시피 데이터를 가져오기 위해, SQLAlchemy를 사용해 데이터베이스와 연결하고 쿼리를 실행합니다. 사용자가 입력한 surveyId와 recipeId를 이용해 해당 데이터를 검색합니다.

\`\`\`python
  # 설문 데이터 가져오기
  survey_data = db.execute(
      text("SELECT health, eating_habits, excluded_ingredients FROM omjm.survey WHERE id = :id"),
      {"id": surveyId}
  ).fetchone()

  # 레시피 데이터 가져오기
  recipe_data = db.execute(
      text("SELECT title, ingredients, instructions FROM omjm.recipe WHERE id = :id"),
      {"id": recipeId}
  ).fetchone()
\`\`\`

이 쿼리들은 각각 surveyId와 recipeId에 해당하는 설문 및 레시피 데이터를 가져옵니다. 가져온 데이터는 나중에 가이드를 생성하는 데 사용됩니다.

### 프롬프트 탬플릿 생성

이제 설문 데이터와 레시피 데이터를 바탕으로, OpenAI API를 통해 가이드를 생성할 수 있는 프롬프트 템플릿을 만듭니다. 프롬프트 템플릿은 사용자 맞춤형 건강 가이드를 생성하는 데 필요한 구조와 내용을 정의합니다.

\`\`\`python
prompt_template = PromptTemplate.from_template(
"""
사용자 입력사항 :
- 가진 질병: {health}
- 원하는 라이프스타일: {eating_habits}
- 제외하고 싶거나 알러지가 있는 재료: {excluded_ingredients}

레시피 정보:
- 이름: {recipe_title}
- 재료: {recipe_ingredients}
- 조리 방법: {recipe_instructions}

관련 건강 정보:
{relevant_info}

위 정보를 바탕으로, 다음 지침에 따라 사용자를 위한 맞춤형 건강 가이드를 작성해주세요:

1. 개인화: 사용자의 건강 상태, 식습관, 제공된 레시피 정보를 바탕으로 과학적이고 개인화된 건강 팁을 작성하세요.
4. 부정문장 삼가 : 적합하지 않다는 말보다는 대체재료를 추천하세요.
5. [],"" 등의 특수 문자는 사용하지 마세요. 팁이 없는 경우 해당 항목을 완전히 생략하고, "팁이 없습니다"와 같은 문구를 출력하지 마세요.
6. 만약 대체재료를 추천할 것이라면 명확한 재료를 추천해주어야 합니다.

조언은 다음과 같은 형식으로 작성해주세요.  :

💊 **{health} 환자 tips**
- (관련 팁을 100자 내외로 작성해주세요. 질병이 없다면 팁을 주지마세요.)
🏋️ **{eating_habits} 관련 tips**
- (관련 팁을 100자 내외로 작성해주세요. 라이프스타일이 없다면 팁을 주지 마세요)
"""
\`\`\`

이 프롬프트 템플릿은 사용자의 설문 데이터와 레시피 정보를 기반으로 맞춤형 건강 가이드를 생성합니다. 예를 들어, health, eating_habits, excluded_ingredients 등의 정보를 통해 사용자의 상태와 요구 사항에 맞는 조언을 제공합니다.

### OpenAI를 통한 가이드 생성

프롬프트 템플릿을 기반으로 OpenAI API에 요청을 보내, 맞춤형 건강 가이드를 생성합니다.

create_prompt 함수는 사용자가 입력한 데이터와 검색된 문서를 기반으로, 최종적으로 제공될 가이드를 생성합니다. 생성된 가이드는 FastAPI를 통해 JSON 형식으로 클라이언트에게 반환됩니다.
이와 같은 방식으로, FastAPI를 활용한 Self-Query RAG의 구현은 매우 유연하고 강력한 검색 기능을 제공합니다. FastAPI는 Python 기반의 웹 애플리케이션 프레임워크로, 성능이 우수하고 간편한 사용성을 갖추고 있어, 다양한 AI 기반 애플리케이션에 적합한 선택입니다

\`\`\`python
def create_prompt(survey_data, recipe_data, relevant_info):
    prompt = prompt_template.format(
        health=survey_data['health'],
        eating_habits=survey_data['eating_habits'],
        excluded_ingredients=survey_data['excluded_ingredients'],
        recipe_title=recipe_data['title'],
        recipe_ingredients=recipe_data['ingredients'],
        recipe_instructions=recipe_data['instructions'],
        relevant_info=relevant_info
    )

    chat_model = ChatOpenAI(api_key=ai, model_name='gpt-4o-mini')

    response = chat_model(prompt)

    return response.content
\`\`\`

이 함수는 생성된 프롬프트를 OpenAI 모델에 전달하고, 반환된 결과를 맞춤형 가이드로 출력합니다. 이 가이드는 사용자의 설문 데이터와 레시피 데이터를 바탕으로 한 매우 개인화된 조언을 제공합니다.

이렇게 프롬프트 템플릿과 OpenAI API를 활용하여 맞춤형 가이드를 생성하는 프로세스가 완성됩니다. 이 과정은 FastAPI로 구현된 API 엔드포인트를 통해 호출되며, 결과는 사용자가 필요로 하는 형태로 전달됩니다.

### ngrok을 사용한 로컬 서버 노출

현재 FastAPI로 구현한 서버는 로컬 환경에서 실행되기 때문에, 이후 이를 외부에서 접근할 수 있도록 하기 위해 ngrok을 사용했습니다. ngrok을 통해 FastAPI 서버를 외부에 노출시켜 Edge Functions와 통합할 수 있었습니다.

\`\`\`
	ngrok http 8000
\`\`\`

굳이 Edge Functions으로 api를 이동시켜야했던 이유와 ngrok 사용에 대한 자세한 방법은 다음 블로그에 하도록 하겠습니다.
`;export{n as default};
