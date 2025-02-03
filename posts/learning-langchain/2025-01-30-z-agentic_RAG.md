---
title: "Agent 활용 | 문서검색과 웹검색을 활용한 Agentic RAG"
---

# Agentic RAG

이번엔 문서 검색을 통해 최신 정보에 근접하여 검색 결과를 가지고 답변을 생성하는 에이전트를 만들어보자.

질문에 따라 문서를 검색하여 답변하거나, 인터넷 검색 도구를 활용하여 답변하는 에이전트를 만들어 볼것이다.

Agent를 활용하여 RAG를 수행한다면 이를 Agentic RAG라고 한다.

## Tools

Agent가 활용할 도구를 정의하여 Agent가 추론(reasoning)을 수행할 때 활용하도록 만들 수 있다.

Tavily Search 는 대표적인 인터넷 검색 도구이며, 검 색을 통해 최신 정보를 얻어 답변을 생성할수 있다. 도구는 검색 뿐만아니라 Python코드를 실행할 수 있는 도구, 직접 정의한 함수를 실행하는 도구 등 다양한 종류와 방법론을 제공한다.

### Web Search Tool : Tavily Search

Langchain에는 Tavily 검색 엔진을 도구로 쉽게 사용할수 있는 내장 도구가 있다.

Tavily Search를 사용하기 위해서는 API key가 필요하다.

- [Tavily API key 발급](https://app.tavily.com/home?code=qBjTyMCT1oPN5i7kReMKW9EhqDEmYgTkswdl_lze12AR_&state=eyJyZXR1cm5UbyI6Ii9ob21lIn0)

발급 받은 API KEY 를 `.env` 파일에 다음과 같이 등록한다.

`TAVILY_API_KEY=(발급 받은 Tavily API KEY)`

```python
from dotenv import load_dotenv
load_dotenv()
```

    True

```python
from langchain_teddynote import logging

logging.langsmith("agentic-rag")
```

    LangSmith 추적을 시작합니다.
    [프로젝트명]
    agentic-rag

```python
# TavilySearchResults 클래슬을 langchain_community.tools.tavily_search 모듈에서 가져온다.
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래슬을 사용하여 도구를 생성한다.
tavily_search_tool = TavilySearchResults(k=6)   # k = 검색 결과 갯수
```

`search.invoke` 함수는 주어진 문자열에 대한 검색을 실행하고,

`invoke()` 함수는 도구를 호출하여 검색을 수행하고 결과를 반환한다.

```python
tavily_search_tool.invoke("판교 네이버 본사의 전화번호는 무엇인가요?")
```

    [{'url': 'https://www.navercorp.com/naver/naverContact',
      'content': '네이버(주) 본사, 주요 계열사 위치 안내 l NAVER Corp. 주요 서비스 요약 기술 주요 기술 요약 AI 기술 검색 기술 로봇 기술 기술 상생 주요 ESG 프로그램 요약 주요 IR 요약 IR 자료실 주요 서비스 요약 기술 주요 기술 요약 AI 기술 검색 기술 로봇 기술 기술 상생 주요 ESG 프로그램 요약 주요 IR 요약 IR 자료실 경기도 성남시 분당구 정자일로 95, NAVER 1784 (우)13561 NAVER 1784, 95, Jeongjail-ro, Bundang-gu, Seongnam-si, Gyeonggi-do, Republic of Korea NAVER Green Factory, 6, Buljeong-ro, Bundang-gu, Seongnam-si, Gyeonggi-do, Republic of Korea 경기도 성남시 분당구 정자일로 95, 네이버 1784 (우)13561 NAVER 1784, 95, Jeongjail-ro, Bundang-gu, Seongnam-si, Gyeonggi-do, Republic of Korea 경기도 성남시 분당구 정자일로 95, NAVER 1784 (우)13561 NAVER 1784, 95, Jeongjail-ro, Bundang-gu, Seongnam-si, Gyeonggi-do, Republic of Korea'},
     {'url': 'https://moonwalker.tistory.com/entry/네이버-고객센터-전화번호-상담원연결-채팅상담-이메일문의-총정리',
      'content': '네이버 고객센터 대표번호는 ☎️1588-3820입니다. 영업시간은 오전 9시부터 오후 6시까지 이며 전화연결 후 1 ~ 5번까지 제공되는 서비스는 아래와 같습니다.'},
     {'url': 'https://d13.aptbusan.co.kr/690',
      'content': '네이버 고객센터 전화번호 네이버 고객센터에 직접 전화를 걸고 상담원과 연결하기 위해서는 특정 전화번호를 이용해야 합니다. 고객센터의 전화번호는 다음과 같습니다: 전화번호: 1588-3830 전화 이용 시 유의사항 전화 상담을 이용할 때는 다음과 같은 사항을 고려하세요: 통화 요금: 일반 전화'},
     {'url': 'https://2ward.tistory.com/38',
      'content': "네이버 고객센터의 위치는 '강원도 춘천시 퇴게로 89 강원전문건설회관'입니다. 직접 가실 일이 있으시다면 참고하시기 바랍니다. NHN 네이버 본사 위치는 '경기도 성남시 분당구 정자동 178-1 경기도 성남시 분당구 불정로 6 NAVER그린팩토리'입니다."},
     {'url': 'https://infonara.tistory.com/141',
      'content': '1. 네이버 고객센터 전화번호는 무엇인가요? 네이버 고객센터 전화번호는 1588-9292입니다. 이 번호로 직접 상담원과 연결할 수 있습니다.'}]

### 문서 기반 검색 도구 : Retriever

우리가 가진 데이터에 대해 조회를 수행할 retriever도 생성한다.

**실습에 활용한 문서**
소프트웨어정책연구소(SPRi) - 2023년 12월호

- 저자: 유재흥(AI정책연구실 책임연구원), 이지수(AI정책연구실 위촉연구원)
- 링크: https://spri.kr/posts/view/23669
- 파일명: SPRI_AI_Brief_2023년12월호\_F.pdf

아래 코드는 웹 기반 문서 로더, 문서 분할기, 벡터 저장소, 그리고 OpenAI 임베딩을 사용하여 문서 검색 시스템을 구축한다.

여기서 PDF문서를 `FAISS` DB에 저장하고 조회하는 Retriever를 생성한다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# PDF 파일 로드
loader = PyPDFLoader("docs/SPRI_AI_Brief_2023년12월호_F.pdf")

# split PDF 파일
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서 분할 및 로드
split_docs = loader.load_and_split(text_splitter)
```

```python
# ! pip install -q transformers torch einops
# ! pip install -q 'numpy<2'
```

```python
from langchain_community.embeddings import JinaEmbeddings
import os

model_name = "jinaai/jina-embeddings-v3"

embeddings_model = JinaEmbeddings(
    model=model_name,
    jina_api_key=os.environ["JINA_API_KEY"]
)
embeddings_model
```

    JinaEmbeddings(session=<requests.sessions.Session object at 0x3009c2390>, model_name='jina-embeddings-v2-base-en', jina_api_key=SecretStr('**********'))

```python
# ! pip install faiss-cpu
```

```python
from langchain_community.vectorstores import FAISS

# 벡터 저장소 생성
faiss_vectorstore = FAISS.from_documents(split_docs, embeddings_model)

# openai 임베딩을 이용한 백터저장소
# vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# retriever 생성
retriever = faiss_vectorstore.as_retriever()
```

이 함수는 retriever 객체의 invoke() 를 사용하여 사용자의 질문에 대한 가장 관련성 높은 문서 를 찾는 데 사용된다.

```python
retriever.invoke("삼성전자가 개발한 생성형 AI 관련 내용을 문서에서 찾아줘")
```

    [Document(id='e7df73e1-29b0-45e2-bb5c-21bd6ae5143a', metadata={'source': 'docs/SPRI_AI_Brief_2023년12월호_F.pdf', 'page': 16, 'page_label': '17'}, page_content='노래나 목소리를 모방한 AI 생성 음악에 대하여 삭제를 요청할 수 있는 기능도 도입할 방침☞ 출처 : Youtube, Our approach to responsible AI innovation, 2023.11.14.'),
     Document(id='0be8b018-9e03-4bd9-af0d-29e780da3dee', metadata={'source': 'docs/SPRI_AI_Brief_2023년12월호_F.pdf', 'page': 4, 'page_label': '5'}, page_content='생성 콘텐츠를 식별할 수 있도록 워터마크를 비롯하여 기술적으로 가능한 기법으로 신뢰할 수 있는 콘텐츠 인증과 출처 확인 메커니즘을 개발 및 구축 ∙사회적 위험과 안전·보안 문제를 완화하는 연구와 효과적인 완화 대책에 우선 투자하고, 기후 위기 대응, 세계 보건과 교육 등 세계적 난제 해결을 위한 첨단 AI 시스템을 우선 개발∙국제 기술 표준의 개발 및 채택을 가속화하고, 개인정보와 지식재산권 보호를 위해 데이터 입력과 수집 시 적절한 보호 장치 구현☞ 출처: G7, Hiroshima Process International Code of Conduct for Advanced AI Systems, 2023.10.30.'),
     Document(id='4562676a-3b0b-4df5-8a14-e567818f4ca2', metadata={'source': 'docs/SPRI_AI_Brief_2023년12월호_F.pdf', 'page': 20, 'page_label': '21'}, page_content='<AI 기술 유형 평균 기술 대비 갖는 임금 프리미엄>'),
     Document(id='855deb33-d045-4229-8ba3-b5fb43de12de', metadata={'source': 'docs/SPRI_AI_Brief_2023년12월호_F.pdf', 'page': 22, 'page_label': '23'}, page_content='홈페이지 : https://spri.kr/보고서와 관련된 문의는 AI정책연구실(jayoo@spri.kr, 031-739-7352)으로 연락주시기 바랍니다.')]

이제 우리가 검색을 수행할 인덱스를 채웠으므로, 이를 에이전트가 제대로 사용할 수 있는 도구로 쉽게 변환할수 있다.

`create_retriever_tool` 함수로 `retriever` 를 도구로 변환한다.

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name = "pdf_search", # 도구 이름입력
    description = "use this tool to search information from the PDF documents." # 도구에 대한 자세한 설명을 영어로 작성
)
retriever_tool

```

    Tool(name='pdf_search', description='use this tool to search information from the PDF documents.', args_schema=<class 'langchain_core.tools.retriever.RetrieverInput'>, func=functools.partial(<function _get_relevant_documents at 0x1090d7600>, retriever=VectorStoreRetriever(tags=['FAISS', 'JinaEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x368941f70>, search_kwargs={}), document_prompt=PromptTemplate(input_variables=['page_content'], input_types={}, partial_variables={}, template='{page_content}'), document_separator='\n\n', response_format='content'), coroutine=functools.partial(<function _aget_relevant_documents at 0x10943e340>, retriever=VectorStoreRetriever(tags=['FAISS', 'JinaEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x368941f70>, search_kwargs={}), document_prompt=PromptTemplate(input_variables=['page_content'], input_types={}, partial_variables={}, template='{page_content}'), document_separator='\n\n', response_format='content'))

### Agent가 사용할 도구목록 정의

웹검색과 문서 검색 두가지의 도구를 사용하여, agent가 사용할수 있도록 도구목록을 정의한다.

```python
# tools 리스트에 tavily_search_tool과 retriever_tool을 추가한다.
tools = [tavily_search_tool, retriever_tool]
```

## Agent 생성

이제 도구 정의까지 되었으니, 에이전트를 생성하여 사용할 수 있다.

먼저, Agent가 활용할 LLM을 정의하고, Agent가 참고할 Prompt를 정의한다.

- 멀티턴 대화를 지원하지 않는다면 "chat_history" 파라미터를 제거한다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.65)

# prompt 정의
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant."
        "Make sure to use the `pdf_search` tool for searching information from the PDF documents."
        "If you can't find the information from the PDF documents, use the `search` tool for searching information from the web."
    ),
    ("placeholder","{chat_history}"), # 대화 기록
    ("human","{input}"), # 사용자 입력
    ("placeholder","{agent_scratchpad}") # 에이전트의 메모장 같은 곳
])
```

다음으로 Tool Calling Agent를 생성한다.

```python
from langchain.agents import create_tool_calling_agent

# tool calling agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)
```

마지막으로, 생성한 Agent를 실행하는 AgentExecutor를 생성한다.

```python
from langchain.agents import AgentExecutor

# agent executor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
)
```

## 에이전트 실행하기

이제 몇 가지 질의에 대해 에이전트를 실행할 수 있다.

현재 이러한 모든 질의는 상태(Stateless) 가 없는 질의입니다(이전 대화내용을 기억하지 않는다).

agent_executor 객체의 invoke 메소드는 딕셔너리 형태의 인자를 받아 처리한다.

```python
from langchain_teddynote.messages import AgentStreamParser

# 각 단계 별 출력을 위한 파서 생성
agent_stream_parser = AgentStreamParser()
```

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
result = agent_executor.stream(
    {"input": "2024년 프로야구 플레이오프 진출한 5개 팀을 검색하여 알려주세요."}
)

for step in result:
    # 중간 단계를 parser 를 사용하여 단계별로 출력
    agent_stream_parser.process_agent_steps(step)
```

    [도구 호출]
    Tool: tavily_search_results_json
    query: 2024년 프로야구 플레이오프 진출 팀
    Log:
    Invoking: `tavily_search_results_json` with `{'query': '2024년 프로야구 플레이오프 진출 팀'}`



    [관찰 내용]
    Observation: [{'url': 'https://allgreat.tistory.com/321', 'content': '2024 프로야구 한국시리즈 포스트시즌 일정 진출팀 / 5위 결정전 경기 타이브레이크, 플레이오프 날짜 순위 위 프로야구 순위는 2024년 9월 29일 기준이다. 6.(일) 준플레이오프 2차전 준플레이오프(준PO) 1~2차전은 정규리그 3위 팀인 LG(엘지)의 홈구장인 잠실야구장에서 열린다. 1차전 : 2024년 10월 13일(일) 2차전 : 2024년 10월 14일(월) 3차전 : 2024년 10월 16일(수) 4차전 : 2024년 10월 17일(목) 5차전 : 2024년 10월 19일(토) 5판 3선승제로 치러지는 프로야구 플레이오프(준결승전) 1~2차전은 정규리그 2위팀인 삼성 라이온즈의 홈구장인 대구 라이온즈파크에서 열린다. 1차전 : 10월 21일(월)\xa0 3차전 : 10월 24일(목) 4차전 : 10월 25일(금) 5차전 : 10월 27일(일) 6차전 : 10월 28일(월) 7차전 : 10월 29일(화) 한국시리즈(코리아시리즈 결승전)는 7전 4선승제로 치러지며, 정규리그 1위팀 기아와 플레이오프 승리팀이 마지막 우승을 놓고 치열한 경기가 펼쳐질 예정이다.'}, ...]
    [최종 답변]
    2024년 프로야구 플레이오프에 진출한 5개 팀은 다음과 같습니다:

    1. **기아 타이거즈** (정규리그 1위)
    2. **삼성 라이온즈** (정규리그 2위)
    3. **LG 트윈스** (정규리그 3위)
    4. **두산 베어스** (정규리그 4위)
    5. **KT 위즈** (정규리그 5위)

    플레이오프는 정규리그 3위 팀인 LG 트윈스와 4위 팀인 두산 베어스 간의 준플레이오프에서 시작되며, 승리한 팀이 KBO 한국시리즈에 진출하게 됩니다.

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
result = agent_executor.stream(
    {"input": "삼성전자가 자체 개발한 생성형 AI는 어떤건가요? 문서에서 찾아주세요."}
)

for step in result:
    # 중간 단계를 parser 를 사용하여 단계별로 출력
    agent_stream_parser.process_agent_steps(step)
```

    [도구 호출]
    Tool: pdf_search
    query: 삼성전자 생성형 AI
    Log:
    Invoking: `pdf_search` with `{'query': '삼성전자 생성형 AI'}`



    [관찰 내용]
    Observation: 노래나 목소리를 모방한 AI 생성 음악에 대하여 삭제를 요청할 수 있는 기능도 도입할 방침☞ 출처 : Youtube, Our approach to responsible AI innovation, 2023.11.14.

    <AI 기술 유형 평균 기술 대비 갖는 임금 프리미엄>

    후 해당 모델이 배포된 타국의 정부 및 연구소와 평가 결과를 공유하고, 학계와 대중이 AI 시스템의 피해와 취약점을 보고할 수 있는 명확한 절차를 수립☞ 출처 : Gov.uk, Introducing the AI Safety Institute, 2023.11.02.             Venturebeat, Researchers turn to Harry Potter to make AI forget about copyrighted material, 2023.10.06.

    생성 콘텐츠를 식별할 수 있도록 워터마크를 비롯하여 기술적으로 가능한 기법으로 신뢰할 수 있는 콘텐츠 인증과 출처 확인 메커니즘을 개발 및 구축 ∙사회적 위험과 안전·보안 문제를 완화하는 연구와 효과적인 완화 대책에 우선 투자하고, 기후 위기 대응, 세계 보건과 교육 등 세계적 난제 해결을 위한 첨단 AI 시스템을 우선 개발∙국제 기술 표준의 개발 및 채택을 가속화하고, 개인정보와 지식재산권 보호를 위해 데이터 입력과 수집 시 적절한 보호 장치 구현☞ 출처: G7, Hiroshima Process International Code of Conduct for Advanced AI Systems, 2023.10.30.
    [도구 호출]
    Tool: tavily_search_results_json
    query: 삼성전자 생성형 AI
    Log:
    Invoking: `tavily_search_results_json` with `{'query': '삼성전자 생성형 AI'}`



    [관찰 내용]
    Observation: [{'url': 'https://www.yna.co.kr/view/AKR20231108041400003', 'content': "삼성전자[005930]가 자체 개발한 생성형 인공지능(AI) 모델 '삼성 가우스'(Samsung Gauss)가 처음 공개됐다. 삼성전자는 삼성 가우스를 활용해 임직원의 업무 생산성을 높이는 한편, 생성형 AI 모델을 단계적으로 제품에 탑재해 새로운 사용자 경험을 제공한다는 계획이다."}, {'url': 'https://zdnet.co.kr/view/?no=20231108081251', 'content': "삼성전자가 자체 개발한 생성형 AI 모델 '삼성 가우스(Samsung Gauss)'를 최초로 공개했다. 삼성전자는 가우스를 활용해 회사 내 업무 혁신을 추진하고"},...]
    [최종 답변]
    삼성전자가 자체 개발한 생성형 AI 모델은 **삼성 가우스(Samsung Gauss)**입니다. 이 모델은 클라우드와 온디바이스를 위한 여러 생성형 AI 모델이 결합된 초거대 생성형 AI 모델 패밀리로, 생성형 AI 모델과 어시스턴트로 구성되어 있습니다. 삼성 가우스는 임직원의 업무 생산성을 높이는 데 사용되며, 단계적으로 제품에 탑재하여 새로운 사용자 경험을 제공할 계획입니다.

    삼성 가우스는 다양한 기능을 제공하는 대화형 AI 서비스인 **삼성 가우스 포탈(Samsung Gauss Portal)**을 통해 문서 요약, 번역, 메일 작성 등과 같은 사무 업무를 효율적으로 처리할 수 있도록 지원하고 있습니다. 삼성전자는 이 모델을 통해 사내 생산성을 향상시키고, 고객에게 차별화된 경험을 제공할 예정입니다.

    최근에는 후속 모델인 **삼성 가우스2(Samsung Gauss2)**도 공개되어, 향상된 성능과 효율을 바탕으로 다양한 활용 방안이 설명될 예정입니다. 이를 통해 코드 아이 서비스의 성능 개선과 멀티모달 기능 지원 등의 계획이 포함되어 있습니다.

## 이전 대화 내용을 기억하는 Agent : RunnableWithMessageHistory

이전의 대화내용을 기억하기 위해서는 `RunnableWithMessageHistory` 를 사용하여 `AgentExecutor` 를 감싸준다.

[RunnableWithMessageHistory 참고 문서](https://wikidocs.net/254682)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# session_id를 저장할 딕셔너리
store = {}

# sessionId를 기반으로 세션기록을 가져오는 함수

def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

# 메시지 기록이 추가된 에이전트 생성
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history, # 대화 session_id
    input_messages_key="input", # 프롬프트의 질문이 입력되는 key: "input"
    history_messages_key="chat_history" # 프롬프트의 메시지가 입력되는 key: "chat_history"
)
```

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
response = agent_with_history.stream(
    {"input": "삼성전자가 자체 개발한 생성형 AI는 어떤건가요? 문서에서 찾아주세요."},
    config={"configurable": {"session_id": "1234abcd"}}
)

# 중간 단계를 parser 를 사용하여 단계별로 출력
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [도구 호출]
    Tool: pdf_search
    query: 삼성전자 생성형 AI
    Log:
    Invoking: `pdf_search` with `{'query': '삼성전자 생성형 AI'}`



    [관찰 내용]
    Observation: 노래나 목소리를 모방한 AI 생성 음악에 대하여 삭제를 요청할 수 있는 기능도 도입할 방침☞ 출처 : Youtube, Our approach to responsible AI innovation, 2023.11.14.

    <AI 기술 유형 평균 기술 대비 갖는 임금 프리미엄>

    후 해당 모델이 배포된 타국의 정부 및 연구소와 평가 결과를 공유하고, 학계와 대중이 AI 시스템의 피해와 취약점을 보고할 수 있는 명확한 절차를 수립☞ 출처 : Gov.uk, Introducing the AI Safety Institute, 2023.11.02.             Venturebeat, Researchers turn to Harry Potter to make AI forget about copyrighted material, 2023.10.06.

    생성 콘텐츠를 식별할 수 있도록 워터마크를 비롯하여 기술적으로 가능한 기법으로 신뢰할 수 있는 콘텐츠 인증과 출처 확인 메커니즘을 개발 및 구축 ∙사회적 위험과 안전·보안 문제를 완화하는 연구와 효과적인 완화 대책에 우선 투자하고, 기후 위기 대응, 세계 보건과 교육 등 세계적 난제 해결을 위한 첨단 AI 시스템을 우선 개발∙국제 기술 표준의 개발 및 채택을 가속화하고, 개인정보와 지식재산권 보호를 위해 데이터 입력과 수집 시 적절한 보호 장치 구현☞ 출처: G7, Hiroshima Process International Code of Conduct for Advanced AI Systems, 2023.10.30.
    [최종 답변]
    삼성전자가 개발한 생성형 AI와 관련된 정보는 다음과 같습니다:

    1. **AI 생성 음악**: 삼성전자는 노래나 목소리를 모방한 AI 생성 음악에 대해 삭제 요청할 수 있는 기능을 도입할 방침입니다.

    2. **신뢰할 수 있는 콘텐츠 인증**: 생성 콘텐츠를 식별할 수 있도록 워터마크를 비롯한 기술적 기법을 개발하고, 신뢰할 수 있는 콘텐츠 인증 및 출처 확인 메커니즘을 구축하고 있습니다.

    3. **사회적 위험 완화**: 삼성전자는 사회적 위험 및 안전·보안 문제를 완화하기 위한 연구와 효과적인 대책에 우선 투자하고 있습니다.

    4. **글로벌 문제 해결**: 기후 위기 대응, 세계 보건과 교육 등 세계적 난제 해결을 위한 첨단 AI 시스템을 우선 개발하고 있습니다.

    5. **데이터 보호**: 개인정보와 지식재산권 보호를 위해 데이터 입력과 수집 시 적절한 보호 장치를 구현할 계획입니다.

    이 정보는 삼성전자의 AI 기술 개발과 관련된 전략을 포함하고 있습니다. 추가적인 세부사항이나 다른 정보가 필요하시면 말씀해 주세요!

```python
response = agent_with_history.stream(
    {"input": "이전의 답변을 영어로 번역해 주세요."},
    # session_id 설정
    config={"configurable": {"session_id": "1234abcd"}},
)

# 출력 확인
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [최종 답변]
    Here is the translation of the previous response into English:

    1. **AI-Generated Music**: Samsung is planning to introduce a feature that allows users to request the deletion of AI-generated music that imitates songs or voices.

    2. **Reliable Content Authentication**: The company is developing technical methods, including watermarks, to identify generated content and establish mechanisms for verifying reliable content and its sources.

    3. **Mitigating Social Risks**: Samsung is prioritizing research and effective measures to mitigate social risks and safety/security issues by investing in them.

    4. **Solving Global Issues**: The company is focusing on developing advanced AI systems to address global challenges such as climate crisis response, global health, and education.

    5. **Data Protection**: Samsung plans to implement appropriate protective measures when inputting and collecting data to safeguard personal information and intellectual property rights.

    This information includes strategies related to Samsung's AI technology development. If you need additional details or other information, please let me know!

## 전체 Agent 템플릿 코드

```python
! pip install -q pymupdf
```

```python
# 필요한 모듈 import
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain_community.embeddings import JinaEmbeddings
import os

########## 1. 도구를 정의합니다 ##########

### 1-1. Search 도구 ###
# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=6은 검색 결과를 6개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=6)

### 1-2. PDF 문서 검색 도구 (Retriever) ###
# PDF 파일 로드. 파일의 경로 입력
loader = PyMuPDFLoader("docs/SPRI_AI_Brief_2023년12월호_F.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)


#### jinaAI embeddings 모델 사용
model_name = "jinaai/jina-embeddings-v3"

embeddings_model = JinaEmbeddings(
    model=model_name,
    jina_api_key=os.environ["JINA_API_KEY"]
)

# VectorStore를 생성합니다.
# vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
vector = FAISS.from_documents(split_docs, embeddings_model)

# Retriever를 생성합니다.
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)

### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]

########## 2. LLM 을 정의합니다 ##########
# LLM 모델을 생성합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

########## 3. Prompt 를 정의합니다 ##########

# Prompt 를 정의합니다 - 이 부분을 수정할 수 있습니다!
# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

########## 4. Agent 를 정의합니다 ##########

# 에이전트를 생성합니다.
# llm, tools, prompt를 인자로 사용합니다.
agent = create_tool_calling_agent(llm, tools, prompt)

########## 5. AgentExecutor 를 정의합니다 ##########

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

# session_id 를 저장할 딕셔너리 생성
store = {}


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

########## 7. Agent 파서를 정의합니다. ##########
agent_stream_parser = AgentStreamParser()
```

```python
# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "구글이 앤스로픽에 투자한 금액을 문서에서 찾아줘"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [도구 호출]
    Tool: pdf_search
    query: 구글 앤스로픽 투자 금액
    Log:
    Invoking: `pdf_search` with `{'query': '구글 앤스로픽 투자 금액'}`



    [관찰 내용]
    Observation: Ⅰ. 인공지능 산업 동향 브리프

    <AI 기술 유형 평균 기술 대비 갖는 임금 프리미엄>

    홈페이지 : https://spri.kr/
    보고서와 관련된 문의는 AI정책연구실(jayoo@spri.kr, 031-739-7352)으로 연락주시기 바랍니다.

    2023년 12월호
    [도구 호출]
    Tool: tavily_search_results_json
    query: 구글 앤스로픽 투자 금액
    Log:
    Invoking: `tavily_search_results_json` with `{'query': '구글 앤스로픽 투자 금액'}`



    [관찰 내용]
    Observation: [{'url': 'https://news.mtn.co.kr/news-detail/2023102914402653731', 'content': '구글, 오픈AI 경쟁자 앤스로픽에 최대 20억 달러 투자 구글, 오픈AI 경쟁자 앤스로픽에 최대 20억 달러 투자 구글이 오픈AI의 경쟁사인 인공지능(AI) 스타트업 앤스로픽에 대규모 자금을 투자한다. 월스트리트저널(WSJ)은 구글이 앤스로픽에 최대 20억 달러(2조7000억원)를 투자한다고 27일(현지시각) 보도했다. WSJ에 따르면 구글은 앤스로픽에 5억달러를 우선 투자했으며, 이후 추가로 15억달러를 투자하기로 합의했다. 구글은 올해 초에도 앤스로픽에 5억5000달러를 투자한 것으로 전해졌다. 앞서 지난달 세계 최대 전자상거래 업체 아마존이 앤스로픽에 최대 40억달러 투자를 결정했다. 아마존은 앤스로픽에 초기 투자금으로 12억5000만달러를 제공하고, 향후 일정 조건에 따라 최대 40억달러까지 투자액을 늘릴 계획을 세운 것으로 알려졌다. 구글은 MS가 선점한 AI 시장에서 추격을 위해 스타트업에 대한 투자를 확대하고 있다. 은주성 MTN 머니투데이방송 기자 MTN Exclusive MTN 영상뉴스 MTN YOUTUBE 제호 : MTN(엠티엔) 주소 : (07328)서울 영등포구 여의나루로 60, 4층 (여의도동, 여의도우체국) 등록번호 : 서울, 아01083'},
    ...,
    'content': '앤스로픽 시장 가치는 올해 초 기준 40억 달러에 달한다. 구글은 MS에 선점당한 AI 시장을 따라잡기 위해 스타트업에 대한 투자를 확대하고 있다. 앤스로픽 외에도 동영상 제작 툴을 만드는 런어웨이(Runway)와 오픈 소스 소프트웨어 서비스 업체인 허깅 페이스(Hugging'}]
    [최종 답변]
    구글은 인공지능 스타트업 앤스로픽에 최대 20억 달러를 투자하기로 했습니다. 초기에는 5억 달러를 투자하고, 이후 추가로 15억 달러를 투자하기로 합의했습니다. 올해 초에도 구글은 앤스로픽에 5억 5천만 달러를 투자한 바 있습니다.

```python
# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 영어로 번역해 주세요"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [최종 답변]
    Google has agreed to invest up to $2 billion in the AI startup Anthropic. Initially, they will invest $500 million, with an agreement to invest an additional $1.5 billion later. Earlier this year, Google also invested $550 million in Anthropic.

```python
# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 SNS 게시글 형태로 100자 내외로 작성하세요."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [최종 답변]
    Google is investing up to $2B in AI startup Anthropic, starting with $500M. Earlier this year, they invested $550M. #AI #Investment

```python
# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 한국어로 답변하세요"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

    [최종 답변]
    구글이 AI 스타트업 앤스로픽에 최대 20억 달러를 투자합니다. 초기 5억 달러 투자 후, 올해 초 5억 5천만 달러를 투자했습니다. #AI #투자
