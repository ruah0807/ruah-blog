const n=`---
title: "Agent 주요기능 | LangChain 도구(tools) 바인딩"
date: 2025-01-28
---

# 도구(tools)

## LLM에 도구를 바인딩하는 방법(Binding Tools to LLM)

LLM이 tool을 호출할수 있기 위해 chat 요청을 할 때 모델에 tool schema를 전달해야한다.
tool calling 기능을 지원하는 Langchain Chat Model은 \`.bind_tools()\` 메서드를 구현하여 LangChain 도구객체, Pydantic 클래스 또는 json schema를 수신하고 공급자별 예상 형식으로 채팅 모델에 바인딩할 수 있다.

바인딩 된 Chat model의 이후 호출은 모델 API에 대한 모든 호출에 tool schema를 포함하여 모델에 전달된다.

1. 도구 정의
2. 도구 바인딩
3. 도구 사용

\`\`\`python
from dotenv import load_dotenv

load_dotenv()

\`\`\`

    True

### LLM에 바인딩할 Tool 정의

tool을 정의

- \`get_word_length\` : 문자열의 길이를 반환하는 함수
- \`add_function\` : 두 숫자를 더하는 함수
- \`naver_news_crawler\` : 네이버 뉴스 크롤링 반환 함수

참고로 도구를 정의할 때 @tool 데코레이터를 사용하여 도구를 정의할 수 있고, docstring은 가급적 영어로 작성하는것을 권장.

\`\`\`python
import re
import requests
from bs4 import BeautifulSoup
from langchain.agents import tool

# define tool
@tool
def get_word_length(text: str) -> int:
    """Returns the number of words in the text"""
    return len(text)

@tool
def add_function(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

@tool
def naver_news_crawler(news_url: str) -> str:
    """Crawls a 네이버 (naver.com) news article and returns the body content."""
    # HTTP GET 요청 보내기
    response = requests.get(news_url)

    # 요청이 성공했는지 확인
    if response.status_code == 200:
        # BeautifulSoup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.text, "html.parser")

        # 원하는 정보 추출
        title = soup.find("h2", id="title_area").get_text()
        content = soup.find("div", id="contents").get_text()
        cleaned_title = re.sub(r"\\n{2,}", "\\n", title)
        cleaned_content = re.sub(r"\\n{2,}", "\\n", content)
    else:
        print(f"HTTP 요청 실패. 응답 코드: {response.status_code}")

    return f"{cleaned_title}\\n{cleaned_content}"

tools = [get_word_length, add_function, naver_news_crawler]
\`\`\`

### bind_tools()로 llm에 도구 바인딩

LLM모델에 \`bind_tools()\` 메서드를 사용하여 도구를 바인딩할 수 있다.

\`\`\`python
from langchain_openai import ChatOpenAI

# create llm model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.65)

# bind tools
llm_with_tools = llm.bind_tools(tools)

\`\`\`

이제 실행결과를 확인하자.
결과는 \`tool_calls\`에 저장이되고, \`.tool_calls\` 속성을 통해 호출 결과를 확인할 수 있다.

- \`name\` : 도구 이름
- \`args\` : 도구 인자

\`\`\`python
# result
# llm_with_tools.invoke("What is the length of the word \`RuahKim\`?").tool_calls
llm_with_tools.invoke("What is the length of the word \`RuahKim\`?").tool_calls


\`\`\`

    [{'name': 'get_word_length',
      'args': {'text': 'RuahKim'},
      'id': 'call_pXOqsWiqxL8fpNIB7rOqkS3t',
      'type': 'tool_call'}]

위의 코드는 결과는 반환이 되지않고 어떤 tool을 호출했는지 확인할 수 있다.

이를 해결하기 위해 \`llm_with_tools\`와 \`JsonOutputToolsParser\`를 연결하여 \`tool_calls\`를 파싱하고 결과를 출력할수있다.

\`\`\`python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# binding tool + parser
chain = llm_with_tools | JsonOutputToolsParser(tools = tools)

# 실행 결과
tool_call_results = chain.invoke("What is the length of the word \`RuahKim\`?")

tool_call_results
\`\`\`

    [{'args': {'text': 'RuahKim'}, 'type': 'get_word_length'}]

여기서 출력된 \`type\`은 호출된 tool(호출함수)이고, \`args\`는 호출된 함수(도구)에 전달되는 인자이다.

\`\`\`python
print(tool_call_results, end="\\n\\n==========\\n\\n")
# 첫 번째 도구 호출 결과
single_result = tool_call_results[0]
# 도구 이름
print(single_result["type"])
# 도구 인자
print(single_result["args"])

\`\`\`

    [{'args': {'text': 'RuahKim'}, 'type': 'get_word_length'}]

    ==========

    get_word_length
    {'text': 'RuahKim'}

도구 이름과 일치하는 도구를 찾아 실행

\`\`\`python
tool_call_results[0]["type"], tools[0].name
\`\`\`

    ('get_word_length', 'get_word_length')

\`\`\`python
def execute_tool_calls(tool_call_results):
    """
    도구 호출 결과를 실행하는 함수
    :param tool_call_results: 도구 호출 결과 리스트
    :param tools: 사용 가능한 도구 리스트
    """
    # Iterate through the list of tool call results
    for tool_call_result in tool_call_results:
        # 도구의 이름과 인자 추출
        tool_name = tool_call_result["type"] # 도구 이름(함수)
        tool_args = tool_call_result["args"] # 도구에 전달되는 인자

        # 도구 이름과 일치하는 도구를 찾아 실행
        # next() 함수를 사용하여 일치하는 첫번째 도구를 찾음
        matching_tool= next((tool for tool in tools if tool.name == tool_name), None)

        if matching_tool:
            # 일치하는 도구를 찾았다면 해당 도구 실행
            result = matching_tool.invoke(tool_args)
            print(f"[도구 이름] {tool_name} [Argument] {tool_args}\\n[결과] {result}")
        else:
            print(f"일치하는 도구를 찾을 수 없습니다: {tool_name}")

execute_tool_calls(tool_call_results)
\`\`\`

    [도구 이름] get_word_length [Argument] {'text': 'RuahKim'}
    [결과] 7

### bind_tools + Parser + Execution 한번에 실행하기

#### 모든 과정을 한번에 실행하는 방법

- \`llm_with_tools\` : 도구 바인딩 모델
- \`JsonOutputToolsParser\` : 도구 호출 결과를 파싱하는 파서
- \`execute_tool_calls\` : 도구 호출 결과를 실행하는 함수

#### Flow

1. 모델에 도구 바인딩
2. 도구 호출 결과를 파싱
3. 도구 호출 결과를 실행

\`\`\`python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

chain = llm_with_tools | JsonOutputToolsParser(tools = tools) | execute_tool_calls
\`\`\`

\`\`\`python
# get_word_length 실행 결과
chain.invoke("What is the length of the word \`RuahKim\`?")

\`\`\`

    [도구 이름] get_word_length [Argument] {'text': 'RuahKim'}
    [결과] 7

\`\`\`python
# add_function 실행 결과
chain.invoke("114.5 + 121.2")
print(114.5 + 121.2)
\`\`\`

    [도구 이름] add_function [Argument] {'a': 114.5, 'b': 121.2}
    [결과] 235.7
    235.7

\`\`\`python
# naver_news_crawler 실행 결과
chain.invoke(
    "뉴스 기사 크롤링해줘: https://n.news.naver.com/mnews/article/008/0005146446"
)
\`\`\`

    [도구 이름] naver_news_crawler [Argument] {'news_url': 'https://n.news.naver.com/mnews/article/008/0005146446'}
    [결과] 1년에 '13억' 번 상위 1% 유튜버들…"계좌로 후원, 꼼수 탈세 막아야"

    [the300]
    (광주=뉴스1) 김태성 기자 = 정일영 더불어민주당 의원이 24일 정부광주지방합동청사에서 열린 국회 기획재정위원회 광주지방국세청 국정감사에서 질의하고 있다.2024.10.24/뉴스1  Copyright (C) 뉴스1. All rights reserved. 무단 전재 및 재배포,  AI학습 이용 금지. /사진=(광주=뉴스1) 김태성 기자유튜버나 BJ(인터넷 방송 진행자) 가운데 상위 1%의 평균 한 해 수입이 13억원이 넘는다는 조사결과가 나왔다. 정일영 더불어민주당 의원이 27일 국세청으로부터 제출받은 자료에 따르면 지난해 유튜버, BJ 등 1인 미디어 창작자가 신고한 '2023년 수입금액'은 총 1조7861억원으로 나타났다. 정 의원실에 따르면 2023년 기준 1인 미디어 창작자로 수입을 신고한 인원은 총 2만4797명이었다. 신고 인원은 2019년 1327명에서 2020년 9449명으로 급증했다. 2021년에는 1만6294명, 2022년에는 1만9290명으로 해마다 늘어나는 추세다.  이들이 신고한 연간 수입금액도 증가 추세다. 총 수입액은 2019년 1011억원에서 2020년 5339억원, 2021년 1조83억원, 2022년 1조4537억원으로 집계됐다. 코로나19(COVID-19) 유행기를 거치며 유튜브와 같은 온라인 영상 플랫폼 시장이 확대되고 1인 미디어 창작자가 증가한 영향으로 해석됐다. 1인 미디어 창작자 수입 상위 1%에 해당하는 247명의 총 수입은 3271억원이었다. 이는 전체 수입의 18.3%다. 1인당 연간 평균 13억2500만원을 번 셈이다. 이는 4년 전 2019년 상위 1%의 평균 수입(978억원)보다 35.5% 늘어난 수치다. 또 1인 미디어 상위 10%인 2479명의 총 수입은 8992억원으로 1인당 평균 수입은 3억6200만원이었다. 정 의원은 "1인 미디어 시장 규모가 커지면서 영상 조회수를 높여 광고, 개인 후원 등 수입을 늘리기 위한 극단적이고 자극적인 콘텐츠 생산이 늘어나고 있다"며 "최근 정치 유튜버를 비롯하여 일반적인 유튜버들 사이에서도 사실을 왜곡하거나 극단적인 표현을 사용하는 행태가 지난 12.3 내란 이후 더욱 늘어나 우려스럽다"고 밝혔다. 이어 "유튜버·BJ 등 연수입이 매년 급격하게 늘어나고 있는데도 불구하고 세무조사 건수는 최근 3년 동안 거의 증가하지 않고 있으므로 강력한 세무조사를 촉구한다"고 했다.정 의원 측은 또 "3억원 이상 수입을 내는 상위 10%의 유튜버들이 동영상 자막 등에 개인계좌를 표기해서 후원을 유도하는 경우 편법적 탈세를 막을 수 없을 것"이라며 "자극적이고 극단적인 콘텐츠 양산을 막기 위해서라도 국세청은 강도 높은 세무조사를 체계적이고 전면적으로 설계해 실시해야 할 것"이라고 밝혔다.
    김성은 기자 (gttsw@mt.co.kr)
    기자 프로필
    머니투데이
    머니투데이
    김성은 기자
    구독
    구독중
    구독자 0
    응원수
    0
    이재명, 첫 고속터미널 귀성인사...셀카 찍고 "잘 다녀오세요"
    '금배지' 뺀 류호정 깜짝 근황…"목수로 취업, 전직과 이직 그만"
    머니투데이의 구독 많은 기자를 구독해보세요!
    닫기
    Copyright ⓒ 머니투데이. All rights reserved. 무단 전재 및 재배포, AI 학습 이용 금지.

    이 기사는 언론사에서 정치 섹션으로 분류했습니다.
    기사 섹션 분류 안내
    기사의 섹션 정보는 해당 언론사의 분류를 따르고 있습니다. 언론사는 개별 기사를 2개 이상 섹션으로 중복 분류할 수 있습니다.
    닫기
    구독
    메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!
    구독중
    메인에서 바로 보는 언론사 편집 뉴스 지금 바로 확인해보세요!
    통찰과 깊이가 다른 국제시사 · 문예 매거진 [PADO]
    QR 코드를 클릭하면 크게 볼 수 있어요.
    QR을 촬영해보세요.
    통찰과 깊이가 다른 국제시사 · 문예 매거진 [PADO]
    닫기
    70돌 된 미피의 생일 기념전에 초대합니다
    QR 코드를 클릭하면 크게 볼 수 있어요.
    QR을 촬영해보세요.
    70돌 된 미피의 생일 기념전에 초대합니다
    닫기
    머니투데이
    머니투데이
    			주요뉴스해당 언론사에서 선정하며 언론사 페이지(아웃링크)로 이동해 볼 수 있습니다.
    전한길 "난 노사모 출신"…노무현재단 "법적 대응"
    "이따 얘기해"…김종민, 예비신부에 '재방료' 들통
    "다른 남자와 잠자리 강요"…이혼 변호사도 당황
    '상습도박' 슈, SES 자료화면서 모자이크 '굴욕'
    집 곳곳에 커플사진?…지상렬, 동거녀 공개
    이 기사를 추천합니다
    기사 추천은 24시간 내 50회까지 참여할 수 있습니다.
    닫기
    쏠쏠정보
    0
    흥미진진
    0
    공감백배
    0
    분석탁월
    0
    후속강추
    0

    모두에게 보여주고 싶은 기사라면?beta
    이 기사를 추천합니다 버튼을 눌러주세요.  집계 기간 동안 추천을 많이 받은 기사는 네이버 자동 기사배열 영역에 추천 요소로 활용됩니다.
    레이어 닫기

    머니투데이 언론사가 직접 선정한 이슈

    이슈
    트럼프 2.0 시대
    트럼프 사인 전 물러선 콜롬비아…백악관 "관세와 제재 보류"
    이슈
    尹대통령 탄핵 심판
    尹 탄핵 정국에도 與 지지율, 민주당과 초접전…박근혜 때와 뭐가 다른가
    이슈
    더 거칠어진 김정은
    트럼프 보란듯 기싸움 택한 김정은…추석 이어 설에 또 미사일 쐈다
    이슈
    가자지구 휴전
    하마스, 이스라엘 여군 인질 4명 석방…적십자사에 넘겨
    이슈
    봉합 안되는 의대증원 갈등
    정부 '당근'에도 전공의 꿈쩍 안 했다…2.2%만 복귀 모집 지원
    이전
    다음
    머니투데이 언론사홈 바로가기

    기자 구독 후 기사보기
    구독 없이 계속 보기

## bind_tools -> Agent & AgentExecutor로 대체

\`bind_tools()\` 는 모델에 사용할 수 있는 스키마(도구)를 제공한다.

AgentExecutor 는 실제로 llm 호출, 올바른 도구로 라우팅, 실행, 모델 재호출 등을 위한 실행 루프를 생성한다.

> Agent 와 AgentExecutor 에 대해서는 다음 장에서 자세히 다루게 될 것이다.

\`\`\`python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# create Agent prompt template
agent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are very poserful assistant, but don't know current events."
    ),
    ("user", "{input}"),
                                        #agent_scratchpad : 모델의 출력을 저장하는 변수(메모장 같은 역할을한다)
     MessagesPlaceholder(variable_name= "agent_scratchpad")
])

# create model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.65)







from langchain.agents import AgentExecutor, create_tool_calling_agent

# 이전에 정의한 도구 사용
tools = [get_word_length, add_function, naver_news_crawler]

# create agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

\`\`\`

\`\`\`python
# get_word_length 실행
result = agent_executor.invoke({"input" : "What is the length of the word \`RuahKim\`?"})

print(result["output"])
\`\`\`

    \x1B[1m> Entering new AgentExecutor chain...\x1B[0m
    \x1B[32;1m\x1B[1;3m
    Invoking: \`get_word_length\` with \`{'text': 'RuahKim'}\`


    \x1B[0m\x1B[36;1m\x1B[1;3m7\x1B[0m\x1B[32;1m\x1B[1;3mThe length of the word "RuahKim" is 7 letters.\x1B[0m

    \x1B[1m> Finished chain.\x1B[0m
    The length of the word "RuahKim" is 7 letters.

\`\`\`python
# add_function 실행
result = agent_executor.invoke({"input" : "114.5 + 121.2 의 계산 결과를 알려줘"})
print("result : ",result)
print("result['output'] : ",result["output"])
\`\`\`

    \x1B[1m> Entering new AgentExecutor chain...\x1B[0m
    \x1B[32;1m\x1B[1;3m
    Invoking: \`add_function\` with \`{'a': 114.5, 'b': 121.2}\`


    \x1B[0m\x1B[33;1m\x1B[1;3m235.7\x1B[0m\x1B[32;1m\x1B[1;3m114.5 + 121.2의 계산 결과는 235.7입니다.\x1B[0m

    \x1B[1m> Finished chain.\x1B[0m
    result :  {'input': '114.5 + 121.2 의 계산 결과를 알려줘', 'output': '114.5 + 121.2의 계산 결과는 235.7입니다.'}
    result['output'] :  114.5 + 121.2의 계산 결과는 235.7입니다.

\`AgentExecutor\` 는 도구를 호출하고 결과를 파싱하는 모든 과정을 자동으로 처리한다.

또한 **한번의 실행으로 끝나는 것이 아닌, 모델이 자신의 결과를 확인하고 다시 자신을 호출하는 과정**을 거쳐 최종응답을 llm으로반환한다.

input > llm > tool_call > 응답 > llm > 최종응답

\`\`\`python
# Atent 실행
# Agent 실행
result = agent_executor.invoke(
    {"input": "114.5 + 121.2 + 34.2 + 110.1 의 계산 결과는?"}
)

# 결과 확인
print(result["output"])
print("==========\\n")
print(114.5 + 121.2 + 34.2 + 110.1)
\`\`\`

    \x1B[1m> Entering new AgentExecutor chain...\x1B[0m
    \x1B[32;1m\x1B[1;3m
    Invoking: \`add_function\` with \`{'a': 114.5, 'b': 121.2}\`


    \x1B[0m\x1B[33;1m\x1B[1;3m235.7\x1B[0m\x1B[32;1m\x1B[1;3m
    Invoking: \`add_function\` with \`{'a': 235.7, 'b': 34.2}\`


    \x1B[0m\x1B[33;1m\x1B[1;3m269.9\x1B[0m\x1B[32;1m\x1B[1;3m
    Invoking: \`add_function\` with \`{'a': 269.9, 'b': 110.1}\`


    \x1B[0m\x1B[33;1m\x1B[1;3m380.0\x1B[0m\x1B[32;1m\x1B[1;3m114.5 + 121.2 + 34.2 + 110.1의 계산 결과는 380.0입니다.\x1B[0m

    \x1B[1m> Finished chain.\x1B[0m
    114.5 + 121.2 + 34.2 + 110.1의 계산 결과는 380.0입니다.
    ==========

    380.0

\`\`\`python
result = agent_executor.invoke(
    {"input" :  "뉴스 기사 크롤링 후 요약해줘: https://n.news.naver.com/mnews/article/008/0005146446"}
)

print(result["output"])

\`\`\`

    \x1B[1m> Entering new AgentExecutor chain...\x1B[0m
    \x1B[32;1m\x1B[1;3m
    Invoking: \`naver_news_crawler\` with \`{'news_url': 'https://n.news.naver.com/mnews/article/008/0005146446'}\`


    \x1B[0m\x1B[38;5;200m\x1B[1;3m1년에 '13억' 번 상위 1% 유튜버들…"계좌로 후원, 꼼수 탈세 막아야"

    [the300]
    (광주=뉴스1) 김태성 기자 = 정일영 더불어민주당 의원이 24일 정부광주지방합동청사에서 열린 국회 기획재정위원회 광주지방국세청 국정감사에서 질의하고 있다.2024.10.24/뉴스1  Copyright (C) 뉴스1. All rights reserved. 무단 전재 및 재배포,  AI학습 이용 금지. /사진=(광주=뉴스1) 김태성 기자유튜버나 BJ(인터넷 방송 진행자) 가운데 상위 1%의 평균 한 해 수입이 13억원이 넘는다는 조사결과가 나왔다. 정일영 더불어민주당 의원이 27일 국세청으로부터 제출받은 자료에 따르면 지난해 유튜버, BJ 등 1인 미디어 창작자가 신고한 '2023년 수입금액'은 총 1조7861억원으로 나타났다. 정 의원실에 따르면 2023년 기준 1인 미디어 창작자로 수입을 신고한 인원은 총 2만4797명이었다. 신고 인원은 2019년 1327명에서 2020년 9449명으로 급증했다. 2021년에는 1만6294명, 2022년에는 1만9290명으로 해마다 늘어나는 추세다.  이들이 신고한 연간 수입금액도 증가 추세다. 총 수입액은 2019년 1011억원에서 2020년 5339억원, 2021년 1조83억원, 2022년 1조4537억원으로 집계됐다. 코로나19(COVID-19) 유행기를 거치며 유튜브와 같은 온라인 영상 플랫폼 시장이 확대되고 1인 미디어 창작자가 증가한 영향으로 해석됐다. 1인 미디어 창작자 수입 상위 1%에 해당하는 247명의 총 수입은 3271억원이었다. 이는 전체 수입의 18.3%다. 1인당 연간 평균 13억2500만원을 번 셈이다. 이는 4년 전 2019년 상위 1%의 평균 수입(978억원)보다 35.5% 늘어난 수치다. 또 1인 미디어 상위 10%인 2479명의 총 수입은 8992억원으로 1인당 평균 수입은 3억6200만원이었다. 정 의원은 "1인 미디어 시장 규모가 커지면서 영상 조회수를 높여 광고, 개인 후원 등 수입을 늘리기 위한 극단적이고 자극적인 콘텐츠 생산이 늘어나고 있다"며 "최근 정치 유튜버를 비롯하여 일반적인 유튜버들 사이에서도 사실을 왜곡하거나 극단적인 표현을 사용하는 행태가 지난 12.3 내란 이후 더욱 늘어나 우려스럽다"고 밝혔다. 이어 "유튜버·BJ 등 연수입이 매년 급격하게 늘어나고 있는데도 불구하고 세무조사 건수는 최근 3년 동안 거의 증가하지 않고 있으므로 강력한 세무조사를 촉구한다"고 했다.정 의원 측은 또 "3억원 이상 수입을 내는 상위 10%의 유튜버들이 동영상 자막 등에 개인계좌를 표기해서 후원을 유도하는 경우 편법적 탈세를 막을 수 없을 것"이라며 "자극적이고 극단적인 콘텐츠 양산을 막기 위해서라도 국세청은 강도 높은 세무조사를 체계적이고 전면적으로 설계해 실시해야 할 것"이라고 밝혔다.
    김성은 기자 (gttsw@mt.co.kr)
    기자 프로필
    머니투데이
    머니투데이
    김성은 기자
    구독
    구독중
    구독자 0
    응원수
    0
    이재명, 첫 고속터미널 귀성인사...셀카 찍고 "잘 다녀오세요"
    '금배지' 뺀 류호정 깜짝 근황…"목수로 취업, 전직과 이직 그만"
    머니투데이의 구독 많은 기자를 구독해보세요!
    닫기
    Copyright ⓒ 머니투데이. All rights reserved. 무단 전재 및 재배포, AI 학습 이용 금지.

    이 기사는 언론사에서 정치 섹션으로 분류했습니다.
    기사 섹션 분류 안내
    기사의 섹션 정보는 해당 언론사의 분류를 따르고 있습니다. 언론사는 개별 기사를 2개 이상 섹션으로 중복 분류할 수 있습니다.
    닫기
    구독
    메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!
    구독중
    메인에서 바로 보는 언론사 편집 뉴스 지금 바로 확인해보세요!
    통찰과 깊이가 다른 국제시사 · 문예 매거진 [PADO]
    QR 코드를 클릭하면 크게 볼 수 있어요.
    QR을 촬영해보세요.
    통찰과 깊이가 다른 국제시사 · 문예 매거진 [PADO]
    닫기
    70돌 된 미피의 생일 기념전에 초대합니다
    QR 코드를 클릭하면 크게 볼 수 있어요.
    QR을 촬영해보세요.
    70돌 된 미피의 생일 기념전에 초대합니다
    닫기
    머니투데이
    머니투데이
    			주요뉴스해당 언론사에서 선정하며 언론사 페이지(아웃링크)로 이동해 볼 수 있습니다.
    전한길 "난 노사모 출신"…노무현재단 "법적 대응"
    "이따 얘기해"…김종민, 예비신부에 '재방료' 들통
    "다른 남자와 잠자리 강요"…이혼 변호사도 당황
    '상습도박' 슈, SES 자료화면서 모자이크 '굴욕'
    집 곳곳에 커플사진?…지상렬, 동거녀 공개
    이 기사를 추천합니다
    기사 추천은 24시간 내 50회까지 참여할 수 있습니다.
    닫기
    쏠쏠정보
    0
    흥미진진
    0
    공감백배
    0
    분석탁월
    0
    후속강추
    0

    모두에게 보여주고 싶은 기사라면?beta
    이 기사를 추천합니다 버튼을 눌러주세요.  집계 기간 동안 추천을 많이 받은 기사는 네이버 자동 기사배열 영역에 추천 요소로 활용됩니다.
    레이어 닫기

    머니투데이 언론사가 직접 선정한 이슈

    이슈
    고려아연 경영권 분쟁
    최윤범 주총 승부수 뒤…MBK "형사고발", 고려아연 "타협하자"
    이슈
    환율 '비상'
    "트럼프 관세폭탄, 오늘은 피했다"…한달 만에 환율 1430원대로 '뚝'
    이슈
    내집 마련 도대체 언제쯤?
    빈집 쌓였는데 또 새집?…지방 '미입주' 공포 커져간다
    이슈
    러시아·우크라전쟁 격화
    러시아, '북한군과 우크라군 맞교환' 젤렌스키 제안에 "할 말 없다"
    이슈
    금투세 폐지
    민주당 "금투세 폐지·가상자산 과세 유예 약속 지킨다"
    이전
    다음
    머니투데이 언론사홈 바로가기

    기자 구독 후 기사보기
    구독 없이 계속 보기
    \x1B[0m\x1B[32;1m\x1B[1;3m기사 요약:

    2023년, 상위 1%의 유튜버와 BJ(인터넷 방송 진행자)들이 평균 연간 수입으로 13억 원 이상을 벌어들이고 있다는 조사 결과가 발표됐다. 정일영 더불어민주당 의원이 국세청에서 받은 자료에 따르면, 지난해 1인 미디어 창작자들이 신고한 총 수입은 1조 7861억 원에 달하며, 신고한 창작자 수는 2만 4797명으로 증가 추세에 있다. 특히, 상위 1%인 247명의 총 수입은 3271억 원으로 전체 수입의 18.3%를 차지하고 있다.

    정 의원은 1인 미디어 시장의 성장에 따라 자극적이고 극단적인 콘텐츠 생산이 증가하고 있으며, 세무조사 건수가 증가하지 않고 있는 점을 우려하고 있다. 그는 국세청이 강력한 세무조사를 실시해야 한다고 강조했다.\x1B[0m

    \x1B[1m> Finished chain.\x1B[0m
    기사 요약:

    2023년, 상위 1%의 유튜버와 BJ(인터넷 방송 진행자)들이 평균 연간 수입으로 13억 원 이상을 벌어들이고 있다는 조사 결과가 발표됐다. 정일영 더불어민주당 의원이 국세청에서 받은 자료에 따르면, 지난해 1인 미디어 창작자들이 신고한 총 수입은 1조 7861억 원에 달하며, 신고한 창작자 수는 2만 4797명으로 증가 추세에 있다. 특히, 상위 1%인 247명의 총 수입은 3271억 원으로 전체 수입의 18.3%를 차지하고 있다.

    정 의원은 1인 미디어 시장의 성장에 따라 자극적이고 극단적인 콘텐츠 생산이 증가하고 있으며, 세무조사 건수가 증가하지 않고 있는 점을 우려하고 있다. 그는 국세청이 강력한 세무조사를 실시해야 한다고 강조했다.
`;export{n as default};
