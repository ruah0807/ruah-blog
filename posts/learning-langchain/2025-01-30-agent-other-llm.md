---
title: "Agent 주요기능 | 다른 LLM 호출해보기"
date: 2025-01-30
---

## OpenAI 외의 도구 호출 Agent 만들기(Tool Calling Agent with others)

OpenAI 외에도 Langchain은 `Anthropic`, `Google Gemini`, `Mistral`, `Together.ai`, `Ollama`과 같은 여러 공급자 구현을 지원한다.

여기서는 다양한 LLM으로 도구 호출 에이전트를 생성하고 실행하는 방법을 살펴보자.

Source : https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/

```python
from dotenv import load_dotenv

load_dotenv()

from langchain_teddynote import logging
logging.langsmith("agent-other-llm")
```

    LangSmith 추적을 시작합니다.
    [프로젝트명]
    agent-other-llm

```python
from langchain.tools import tool
from typing import List, Dict
from langchain_teddynote.tools import GoogleNews

# Define tool
@tool
def search_google_news(query: str)-> List[Dict[str, str]] :
    """Search google news by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


print(f"도구 이름 : {search_google_news.name}")
print(f"도구 설명 : {search_google_news.description}")

```

    도구 이름 : search_google_news
    도구 설명 : Search google news by input keyword

```python
# Define tool
tools = [search_google_news]
```

### Agent 용 프롬프트 생성

- `chat_history` : 이전 대화 내용을 저장하는 변수 (멀티턴을 지원하지 않는다면, 생략 가능합니다.)
- `agent_scratchpad` : 에이전트가 임시로 저장하는 변수
- `input` : 사용자의 입력

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

# create prompt
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공한다 (도구의 이름과 역할을 입력)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that can search google news by input keyword"
     ),
     ("placeholder", "{chat_history}"),
     ("human", "{input}"),
     ("placeholder", "{agent_scratchpad}"),
])
```

### Tool Calling 을 지원하는 LLM List

실습 진행을 위해 아래 내용을 설정한다.

- Anthropic
  - [Anthropic API 키 발급 관련](https://console.anthropic.com/settings/keys)
  - .env 파일 내 ANTHROPIC_API_KEY 에 발급받은 키 설정
- Gemini
  - [Gemini API 키 발급 관련](https://aistudio.google.com/app/apikey?hl=ko)
  - .env 파일 내 GOOGLE_API_KEY 에 발급받은 키 설정
- Together AI
  - [Together AI API 키 발급 관련](https://api.together.ai/signin)
  - .env 파일 내 TOGETHER_API_KEY 에 발급받은 키 설정
- Ollama
  - [Ollama Tool Calling 지원 모델 리스트](https://ollama.com/search?c=tools)
  - [이번 실습에 사용할 llama3.1 모델](https://ollama.com/library/llama3.1)
  - 터미널 창에 ollama pull llama3.1 명령어를 입력하여 모델 다운로드
  - 이전에 Ollama 를 사용하지 않았다면, [Ollama](https://wikidocs.net/233805) 를 참고한다.

langchain-ollama 설치를 한 뒤 진행한다.

```python

# ! pip install -qU langchain-ollama==0.1.3
# ! pip install -U notebook
```

```python
# ! pip install -q langchain_anthropic langchain_google_vertexai langchain_ollama
```

```python
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

# GPT-4o-mini
gpt = ChatOpenAI(model="gpt-4o-mini")

# Claude 3.5 Sonnet
claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Gemini 1.5 Pro
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
# gemini = ChatVertexAI(model="gemini-1.5-pro", temperature=0)

# TogetherAI : Llama-3.1-70B-Instruct-turbo
llama = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)

# # Llama 3.1
ollama = ChatOllama(model="llama3.1", temperature=0)

# # Qwen2.5 7B (한글성능 나쁘지 않음.)
qwen = ChatOllama(model="qwen2.5:latest")
```

각 LLM 기반으로 Agent를 생성한다.

```python
from langchain.agents import create_tool_calling_agent

gpt_agent = create_tool_calling_agent(gpt, tools, prompt)
claude_agent = create_tool_calling_agent(claude, tools, prompt)
gemini_agent = create_tool_calling_agent(gemini, tools, prompt)
llama_agent = create_tool_calling_agent(llama, tools, prompt)
ollama_agent = create_tool_calling_agent(ollama, tools, prompt)
qwen_agent = create_tool_calling_agent(qwen, tools, prompt)

```

gemini는 SDK를 설치해야한다고 하니 그냥 넘어가자.

### AgentExecutor 생성 후ㅜ 실행 및 결과 확인

```python
from langchain.agents import AgentExecutor

# gpt_agent 실행
agent_executor = AgentExecutor(
    agent=gpt_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

result = agent_executor.invoke ({
    "input": "미국 AI 투자와 관련된 뉴스를 검색해주세요."
})

print("Agent 실행 결과 : \n", result["output"])


```

    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `search_google_news` with `{'query': '미국 AI 투자'}`


    [0m[36;1m[1;3m[{'url': 'https://news.google.com/rss/articles/CBMiXkFVX3lxTFA3WU9zaU16TlNPWEVVMU1DaHVGQVhTeXhUNzZjSG5RVnRNbEo0Q2duV2JZdlhLOUdCUi1FVEMxVzZRSHVsX1ZlSTdMWUs2dmtORmFrMnZoYzYtVHBzeFHSAXZBVV95cUxPY24wZ05Td0JqSmRsSUZUOG1NZUtmTEdxTTI2aTV3d1RfbHdoYkFDQ1N2MnUtWWJjZmduTUI5RHBGRUs3NFJfUzd0TEgwSHlLSThneWdqSHNNV0tzcHZaazlFUTB3Rm12SG9UVGNaUGlqRm4zVHZR?oc=5', 'content': "트럼프, 'AI 패권' 야심 드러내...중국 노골적 견제 - YTN"}, {'url': 'https://news.google.com/rss/articles/CBMiUEFVX3lxTE05TlIzaS0wSlN3V3NFZ0xtTnh1aVJYOWdSSVowQlE2YkZaaklTQk5IOF9FTE0xSDZYV25RYkV4dFl5RWVYQmFjbHJKMnV6RVQt0gFTQVVfeXFMTnFTNks0eXlSX0NESm1JX1hoOG1MbnlRcnM0ZjEtTDNOeTdxR0V6RVgxdmZEWXpOWGRNMzhSS1ZzblBzcEYwQl8yYlhQSjJ6bW1NMmc?oc=5', 'content': '트럼프, AI 분야 ‘5천억 달러’ 투자 계획 발표 - VOA Korea'}, {'url': 'https://news.google.com/rss/articles/CBMihAFBVV95cUxQN3FmTE9KZ2llMzVFa0dSQUdySlUyLU9DNTliaUgtN2hNSEd2SkRqYVJkUUoxOVNRbThOZ0dOY2FSZWpaTVZTaE9wTjN0UXZMLS1UcGtJRkxFRlhNVU1aay1PRnJUYXlWRHBvMXdGWHBDY2Zka05YUWRVZU9kYlJqMnNQQ1fSAZgBQVVfeXFMTUY1bzd4TlFIeUNMOWxLY2NGZ3NLcjJnZklxbmN4alUzV19HRW1xQTdYbUdvYlJ6dzc0RndwY0xRYWJhalZOWE5Bb3AxMHhDZHIybWtJRVgtQzhGZzV2OUhEODhRQWlPN25VcDM3NWducG5sV3NmQjdLZXd5UjBYNTUwSi1vQjE0MmJiaXZIVXdnM0lEZExsLUc?oc=5', 'content': '트럼프 “美 AI인프라에 5000억 달러 투자” - 조선일보'}, {'url': 'https://news.google.com/rss/articles/CBMiWkFVX3lxTE1mYm1jSXlwV3hpZGhfRjg4eTF4Q3RwQXFSa0RpNkh2TFk2cUIyUEhEYTl1VWFOeWdyMHllZjMxODEycm1DTmt3ajVYWDhaLVozcUFQYWUxdHREQdIBVEFVX3lxTE5uaWY3LVpGT1MydXJhWjFLTERRcGpLZTJjbW9FNU9PWUtzV0c1bS1NRThsM1FUWHlSNHNkUmtILWU5VFFsSHhkd0w3WWdtQ2RIU21CbA?oc=5', 'content': '"中 \'딥시크\' 보니…" 트럼프 발언에 엔비디아 개미 \'충격\' - 한국경제'}, {'url': 'https://news.google.com/rss/articles/CBMiVkFVX3lxTE56ZXB0bWRUOHRsU2ktUW1fOFgxOWhYTjlGWjczbjVHRG9vOEF2VFY5dUszTTBjVkxfZXJoRGNoTlpVRWZoV00xTFlsRlE0bzFrbmJDVjFn?oc=5', 'content': '트럼프 "AI에 718조 투자"…미∙중 패권 3차대전, 韓엔 호재 - 중앙일보'}][0m[32;1m[1;3m다음은 미국 AI 투자와 관련된 최근 뉴스 기사들입니다:

    1. [트럼프, 'AI 패권' 야심 드러내...중국 노골적 견제 - YTN](https://news.google.com/rss/articles/CBMiXkFVX3lxTFA3WU9zaU16TlNPWEVVMU1DaHVGQVhTeXhUNzZjSG5RVnRNbEo0Q2duV2JZdlhLOUdCUi1FVEMxVzZRSHVsX1ZlSTdMWUs2dmtORmFrMnZoYzYtVHBzeFHSAXZBVV95cUxPY24wZ05Td0JqSmRsSUZUOG1NZUtmTEdxTTI2aTV3d1RfbHdoYkFDQ1N2MnUtWWJjZmduTUI5RHBGRUs3NFJfUzd0TEgwSHlLSThneWdqSHNNV0tzcHZaazlFUTB3Rm12SG9UVGNaUGlqRm4zVHZR?oc=5)

    2. [트럼프, AI 분야 ‘5천억 달러’ 투자 계획 발표 - VOA Korea](https://news.google.com/rss/articles/CBMiUEFVX3lxTE05TlIzaS0wSlN3V3NFZ0xtTnh1aVJYOWdSSVowQlE2YkZaaklTQk5IOF9FTE0xSDZYV25RYkV4dFl5RWVYQmFjbHJKMnV6RVQt0gFTQVVfeXFMTnFTNks0eXlSX0NESm1JX1hoOG1MbnlRcnM0ZjEtTDNOeTdxR0V6RVgxdmZEWXpOWGRNMzhSS1ZzblBzcEYwQl8yYlhQSjJ6bW1NMmc?oc=5)

    3. [트럼프 “美 AI인프라에 5000억 달러 투자” - 조선일보](https://news.google.com/rss/articles/CBMihAFBVV95cUxQN3FmTE9KZ2llMzVFa0dSQUdySlUyLU9DNTliaUgtN2hNSEd2SkRqYVJkUUoxOVNRbThOZ0dOY2FSZWpaTVZTaE9wTjN0UXZMLS1UcGtJRkxFRlhNVU1aay1PRnJUYXlWRHBvMXdGWHBDY2Zka05YUWRVZU9kYlJqMnNQQ1fSAZgBQVVfeXFMTUY1bzd4TlFIeUNMOWxLY2NGZ3NLcjJnZklxbmN4alUzV19HRW1xQTdYbUdvYlJ6dzc0RndwY0xRYWJhalZOWE5Bb3AxMHhDZHIybWtJRVgtQzhGZzV2OUhEODhRQWlPN25VcDM3NWducG5sV3NmQjdLZXd5UjBYNTUwSi1vQjE0MmJiaXZIVXdnM0lEZExsLUc?oc=5)

    4. [\"中 '딥시크' 보니…\" 트럼프 발언에 엔비디아 개미 '충격' - 한국경제](https://news.google.com/rss/articles/CBMiWkFVX3lxTE1mYm1jSXlwV3hpZGhfRjg4eTF4Q3RwQXFSa0RpNkh2TFk2cUIyUEhEYTl1VWFOeWdyMHllZjMxODEycm1DTmt3ajVYWDhaLVozcUFQYWUxdHREQdIBVEFVX3lxTE5uaWY3LVpGT1MydXJhWjFLTERRcGpLZTJjbW9FNU9PWUtzV0c1bS1NRThsM1FUWHlSNHNkUmtILWU5VFFsSHhkd0w3WWdtQ2RIU21CbA?oc=5)

    5. [트럼프 \"AI에 718조 투자\"…미∙중 패권 3차대전, 韓엔 호재 - 중앙일보](https://news.google.com/rss/articles/CBMiVkFVX3lxTE56ZXB0bWRUOHRsU2ktUW1fOFgxOWhYTjlGWjczbjVHRG9vOEF2VFY5dUszTTBjVkxfZXJoRGNoTlpVRWZoV00xTFlsRlE0bzFrbmJDVjFn?oc=5)

    이 기사들은 트럼프 전 대통령의 AI 분야에 대한 투자 계획과 관련된 내용을 다루고 있습니다.[0m

    [1m> Finished chain.[0m
    Agent 실행 결과 :
     다음은 미국 AI 투자와 관련된 최근 뉴스 기사들입니다:

    1. [트럼프, 'AI 패권' 야심 드러내...중국 노골적 견제 - YTN](https://news.google.com/rss/articles/CBMiXkFVX3lxTFA3WU9zaU16TlNPWEVVMU1DaHVGQVhTeXhUNzZjSG5RVnRNbEo0Q2duV2JZdlhLOUdCUi1FVEMxVzZRSHVsX1ZlSTdMWUs2dmtORmFrMnZoYzYtVHBzeFHSAXZBVV95cUxPY24wZ05Td0JqSmRsSUZUOG1NZUtmTEdxTTI2aTV3d1RfbHdoYkFDQ1N2MnUtWWJjZmduTUI5RHBGRUs3NFJfUzd0TEgwSHlLSThneWdqSHNNV0tzcHZaazlFUTB3Rm12SG9UVGNaUGlqRm4zVHZR?oc=5)

    2. [트럼프, AI 분야 ‘5천억 달러’ 투자 계획 발표 - VOA Korea](https://news.google.com/rss/articles/CBMiUEFVX3lxTE05TlIzaS0wSlN3V3NFZ0xtTnh1aVJYOWdSSVowQlE2YkZaaklTQk5IOF9FTE0xSDZYV25RYkV4dFl5RWVYQmFjbHJKMnV6RVQt0gFTQVVfeXFMTnFTNks0eXlSX0NESm1JX1hoOG1MbnlRcnM0ZjEtTDNOeTdxR0V6RVgxdmZEWXpOWGRNMzhSS1ZzblBzcEYwQl8yYlhQSjJ6bW1NMmc?oc=5)

    3. [트럼프 “美 AI인프라에 5000억 달러 투자” - 조선일보](https://news.google.com/rss/articles/CBMihAFBVV95cUxQN3FmTE9KZ2llMzVFa0dSQUdySlUyLU9DNTliaUgtN2hNSEd2SkRqYVJkUUoxOVNRbThOZ0dOY2FSZWpaTVZTaE9wTjN0UXZMLS1UcGtJRkxFRlhNVU1aay1PRnJUYXlWRHBvMXdGWHBDY2Zka05YUWRVZU9kYlJqMnNQQ1fSAZgBQVVfeXFMTUY1bzd4TlFIeUNMOWxLY2NGZ3NLcjJnZklxbmN4alUzV19HRW1xQTdYbUdvYlJ6dzc0RndwY0xRYWJhalZOWE5Bb3AxMHhDZHIybWtJRVgtQzhGZzV2OUhEODhRQWlPN25VcDM3NWducG5sV3NmQjdLZXd5UjBYNTUwSi1vQjE0MmJiaXZIVXdnM0lEZExsLUc?oc=5)

    4. [\"中 '딥시크' 보니…\" 트럼프 발언에 엔비디아 개미 '충격' - 한국경제](https://news.google.com/rss/articles/CBMiWkFVX3lxTE1mYm1jSXlwV3hpZGhfRjg4eTF4Q3RwQXFSa0RpNkh2TFk2cUIyUEhEYTl1VWFOeWdyMHllZjMxODEycm1DTmt3ajVYWDhaLVozcUFQYWUxdHREQdIBVEFVX3lxTE5uaWY3LVpGT1MydXJhWjFLTERRcGpLZTJjbW9FNU9PWUtzV0c1bS1NRThsM1FUWHlSNHNkUmtILWU5VFFsSHhkd0w3WWdtQ2RIU21CbA?oc=5)

    5. [트럼프 \"AI에 718조 투자\"…미∙중 패권 3차대전, 韓엔 호재 - 중앙일보](https://news.google.com/rss/articles/CBMiVkFVX3lxTE56ZXB0bWRUOHRsU2ktUW1fOFgxOWhYTjlGWjczbjVHRG9vOEF2VFY5dUszTTBjVkxfZXJoRGNoTlpVRWZoV00xTFlsRlE0bzFrbmJDVjFn?oc=5)

    이 기사들은 트럼프 전 대통령의 AI 분야에 대한 투자 계획과 관련된 내용을 다루고 있습니다.

이제 다양한 llm을 사용하여 agent를 실행해보자.

다음은 입력받은 llm을 사용하여 Agent를 생성하고 실행하여 결과를 출력하는 함수이다.

```python
def execute_agent(llm, tools, input_text, label):
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    result = executor.invoke({"input": input_text})
    print(f"[ '{label}' 실행 결과 ]")

    if isinstance(result["output"], list) and len(result["output"]) > 0:
        for item in result ["output"]:
            if "text" in item:
                print(item["text"])
    elif isinstance(result["output"], str):
        print(result["output"])
    else:
        print(result["output"])

```

각 LLM 별로 에이전트를 생성, 실행하여 결과를 출력.

```python
query = (
    "AI 투자와 관련된 뉴스를 검색하고, 결과를 Instagram 게시글 형식으로 작성해 주세요."
)
```

```python

# gpt
execute_agent(gpt, tools, query, "gpt")
```

    [ 'gpt' 실행 결과 ]
    🌟 **AI 투자 동향** 🌟

    1️⃣ **MS와 메타**: 메타가 AI에 대한 지속적인 투자를 통해 주가 상승세를 이어가고 있다는 소식입니다! 🤑 [자세히 보기](https://news.google.com/rss/articles/CBMiWkFVX3lxTE8yeEdfOWxJSmtHSUJYS0ZXVDJaU1RWUTNzU01EVkJQaFpOREZNRE95REpjWU9YR3ZJTlAyODJVdzU2djNZSW5kUjRpVkdid3lfYU9USlNlLTRuZ9IBVEFVX3lxTE1xVzJZZ1lRQ3R6UnlwYkkwS1dsSVBWV2o5TmZoTjBDMGtRanZzMENMMlE2bE9PbHpGMjVsQ0ZOYlk1Wk96bWJGMzlkQ3pqVVZvTndmcw?oc=5)

    2️⃣ **트럼프 발 AI 투자**: AI 투자가 한국 기업에 새로운 기회를 제공할 수 있다는 전망이 나오고 있습니다. 전력 솔루션 시장이 주목받고 있네요! 🔍 [자세히 보기](https://news.google.com/rss/articles/CBMiW0FVX3lxTE4zbUlUMW5PZ2xBWVBXQjRrdWM1ZlRlN3BoWE1VRkd3cHRlcTVKSEEtLUZrMWlKc2NkRFY5dmJ3YmlmUTctd2Q4SzNxQnJBZnJMYzJxOVo1WjB1em_SAV5BVV95cUxQWjVzN2t0ejY2akhjZGVGbnpRRWw3ZWUwckFLc3U3NXkwaFcxaTVCRWJHY0M0STNhMU1heURwWmlXa285YnJqa2h5NFdkamVEejhwQnpVNE9uVS1aWk9B?oc=5)

    3️⃣ **ETF와 AI 투자**: 금리 하락기에는 ETF와 AI 투자의 조합으로 전략적 자산 배분 가이드가 강조되고 있습니다! 📊 [자세히 보기](https://news.google.com/rss/articles/CBMiVkFVX3lxTFA4WnRkSE1kZnUteWhMdmJWSWFEWUs3Y0tTTVFjc0M2YmNNOEM5b0RHa3pDN0dneXp6ZG9tcGNiQjMtYXF0X2l6a3VoMWlrTHdSVDgyLXlR?oc=5)

    4️⃣ **애플 주가 급등**: AI 투자에 상대적으로 적게 참여했다는 이유로 애플의 주가가 3% 급등하며 시가총액 1위를 탈환했습니다! 🍏 [자세히 보기](https://news.google.com/rss/articles/CBMiWEFVX3lxTE5IYk80R1pzV084dlpXODE3ZWRLTGFhZ0NaWmJoTGRIVE00OWwzcVBTTElUanJjRmtDMmVZV20xUkVoaEVFRnVxYTV1VFJ4T1B2dkxMM1cwUDQ?oc=5)

    5️⃣ **어닝 서프라이즈 메타**: 메타가 어닝 서프라이즈로 인해 주가가 4% 상승하며 AI 인프라에 대한 지속적인 투자 의사를 밝혔습니다! 💹 [자세히 보기](https://news.google.com/rss/articles/CBMiUkFVX3lxTE5kVkZWUWxKZXRuUzQxRkFOc2pVNzdDMkxFUzFGeE51QUZrb2tMMVo5cUptTExXS01jZEFlTXBQUWlTRXBmYUhfUjNXc24xU1daQkE?oc=5)

    📈 **AI 투자 열풍이 계속되고 있습니다! 어떤 기업이 다음 주식의 주인공이 될지 기대됩니다!**

```python
# claude
# execute_agent(claude, tools, query, "claude")
```

claude는 credit balance 가 부족하여 실행이 안되니 그냥 넘어가자.

```python
# google gemini
execute_agent(gemini, tools, query, "gemini")
```

    ---------------------------------------------------------------------------

    InvalidArgument                           Traceback (most recent call last)

    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:178, in _chat_with_retry.<locals>._chat_with_retry(**kwargs)
        177 try:
    --> 178     return generation_method(**kwargs)
        179 # Do not retry for these errors.


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/ai/generativelanguage_v1beta/services/generative_service/client.py:1143, in GenerativeServiceClient.stream_generate_content(self, request, model, contents, retry, timeout, metadata)
       1142 # Send the request.
    -> 1143 response = rpc(
       1144     request,
       1145     retry=retry,
       1146     timeout=timeout,
       1147     metadata=metadata,
       1148 )
       1150 # Done; return the response.


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/gapic_v1/method.py:131, in _GapicCallable.__call__(self, timeout, retry, compression, *args, **kwargs)
        129     kwargs["compression"] = compression
    --> 131 return wrapped_func(*args, **kwargs)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/retry/retry_unary.py:293, in Retry.__call__.<locals>.retry_wrapped_func(*args, **kwargs)
        290 sleep_generator = exponential_sleep_generator(
        291     self._initial, self._maximum, multiplier=self._multiplier
        292 )
    --> 293 return retry_target(
        294     target,
        295     self._predicate,
        296     sleep_generator,
        297     timeout=self._timeout,
        298     on_error=on_error,
        299 )


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/retry/retry_unary.py:153, in retry_target(target, predicate, sleep_generator, timeout, on_error, exception_factory, **kwargs)
        151 except Exception as exc:
        152     # defer to shared logic for handling errors
    --> 153     _retry_error_helper(
        154         exc,
        155         deadline,
        156         sleep,
        157         error_list,
        158         predicate,
        159         on_error,
        160         exception_factory,
        161         timeout,
        162     )
        163     # if exception not raised, sleep before next attempt


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/retry/retry_base.py:212, in _retry_error_helper(exc, deadline, next_sleep, error_list, predicate_fn, on_error_fn, exc_factory_fn, original_timeout)
        207     final_exc, source_exc = exc_factory_fn(
        208         error_list,
        209         RetryFailureReason.NON_RETRYABLE_ERROR,
        210         original_timeout,
        211     )
    --> 212     raise final_exc from source_exc
        213 if on_error_fn is not None:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/retry/retry_unary.py:144, in retry_target(target, predicate, sleep_generator, timeout, on_error, exception_factory, **kwargs)
        143 try:
    --> 144     result = target()
        145     if inspect.isawaitable(result):


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/timeout.py:130, in TimeToDeadlineTimeout.__call__.<locals>.func_with_timeout(*args, **kwargs)
        128     kwargs["timeout"] = remaining_timeout
    --> 130 return func(*args, **kwargs)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/google/api_core/grpc_helpers.py:174, in _wrap_stream_errors.<locals>.error_remapped_callable(*args, **kwargs)
        173 except grpc.RpcError as exc:
    --> 174     raise exceptions.from_grpc_error(exc) from exc


    InvalidArgument: 400 * GenerateContentRequest.contents[2].parts[0].function_response.name: Name cannot be empty.



    The above exception was the direct cause of the following exception:


    ChatGoogleGenerativeAIError               Traceback (most recent call last)

    Cell In[37], line 2
          1 # google gemini
    ----> 2 execute_agent(gemini, tools, query, "gemini")


    Cell In[33], line 4, in execute_agent(llm, tools, input_text, label)
          2 agent = create_tool_calling_agent(llm, tools, prompt)
          3 executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    ----> 4 result = executor.invoke({"input": input_text})
          5 print(f"[ '{label}' 실행 결과 ]")
          7 if isinstance(result["output"], list) and len(result["output"]) > 0:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/chains/base.py:170, in Chain.invoke(self, input, config, **kwargs)
        168 except BaseException as e:
        169     run_manager.on_chain_error(e)
    --> 170     raise e
        171 run_manager.on_chain_end(outputs)
        173 if include_run_info:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/chains/base.py:160, in Chain.invoke(self, input, config, **kwargs)
        157 try:
        158     self._validate_inputs(inputs)
        159     outputs = (
    --> 160         self._call(inputs, run_manager=run_manager)
        161         if new_arg_supported
        162         else self._call(inputs)
        163     )
        165     final_outputs: Dict[str, Any] = self.prep_outputs(
        166         inputs, outputs, return_only_outputs
        167     )
        168 except BaseException as e:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/agents/agent.py:1624, in AgentExecutor._call(self, inputs, run_manager)
       1622 # We now enter the agent loop (until it returns something).
       1623 while self._should_continue(iterations, time_elapsed):
    -> 1624     next_step_output = self._take_next_step(
       1625         name_to_tool_map,
       1626         color_mapping,
       1627         inputs,
       1628         intermediate_steps,
       1629         run_manager=run_manager,
       1630     )
       1631     if isinstance(next_step_output, AgentFinish):
       1632         return self._return(
       1633             next_step_output, intermediate_steps, run_manager=run_manager
       1634         )


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/agents/agent.py:1332, in AgentExecutor._take_next_step(self, name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager)
       1321 def _take_next_step(
       1322     self,
       1323     name_to_tool_map: Dict[str, BaseTool],
       (...)
       1327     run_manager: Optional[CallbackManagerForChainRun] = None,
       1328 ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
       1329     return self._consume_next_step(
       1330         [
       1331             a
    -> 1332             for a in self._iter_next_step(
       1333                 name_to_tool_map,
       1334                 color_mapping,
       1335                 inputs,
       1336                 intermediate_steps,
       1337                 run_manager,
       1338             )
       1339         ]
       1340     )


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/agents/agent.py:1358, in AgentExecutor._iter_next_step(self, name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager)
       1355     intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
       1357     # Call the LLM to see what to do.
    -> 1358     output = self._action_agent.plan(
       1359         intermediate_steps,
       1360         callbacks=run_manager.get_child() if run_manager else None,
       1361         **inputs,
       1362     )
       1363 except OutputParserException as e:
       1364     if isinstance(self.handle_parsing_errors, bool):


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain/agents/agent.py:581, in RunnableMultiActionAgent.plan(self, intermediate_steps, callbacks, **kwargs)
        573 final_output: Any = None
        574 if self.stream_runnable:
        575     # Use streaming to make sure that the underlying LLM is invoked in a
        576     # streaming
       (...)
        579     # Because the response from the plan is not a generator, we need to
        580     # accumulate the output into final output and return that.
    --> 581     for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
        582         if final_output is None:
        583             final_output = chunk


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:3401, in RunnableSequence.stream(self, input, config, **kwargs)
       3395 def stream(
       3396     self,
       3397     input: Input,
       3398     config: Optional[RunnableConfig] = None,
       3399     **kwargs: Optional[Any],
       3400 ) -> Iterator[Output]:
    -> 3401     yield from self.transform(iter([input]), config, **kwargs)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:3388, in RunnableSequence.transform(self, input, config, **kwargs)
       3382 def transform(
       3383     self,
       3384     input: Iterator[Input],
       3385     config: Optional[RunnableConfig] = None,
       3386     **kwargs: Optional[Any],
       3387 ) -> Iterator[Output]:
    -> 3388     yield from self._transform_stream_with_config(
       3389         input,
       3390         self._transform,
       3391         patch_config(config, run_name=(config or {}).get("run_name") or self.name),
       3392         **kwargs,
       3393     )


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:2188, in Runnable._transform_stream_with_config(self, input, transformer, config, run_type, **kwargs)
       2186 try:
       2187     while True:
    -> 2188         chunk: Output = context.run(next, iterator)  # type: ignore
       2189         yield chunk
       2190         if final_output_supported:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:3351, in RunnableSequence._transform(self, input, run_manager, config, **kwargs)
       3348     else:
       3349         final_pipeline = step.transform(final_pipeline, config)
    -> 3351 yield from final_pipeline


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:1406, in Runnable.transform(self, input, config, **kwargs)
       1403 final: Input
       1404 got_first_val = False
    -> 1406 for ichunk in input:
       1407     # The default implementation of transform is to buffer input and
       1408     # then call stream.
       1409     # It'll attempt to gather all input into a single chunk using
       1410     # the `+` operator.
       1411     # If the input is not addable, then we'll assume that we can
       1412     # only operate on the last chunk,
       1413     # and we'll iterate until we get to the last chunk.
       1414     if not got_first_val:
       1415         final = ichunk


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:5559, in RunnableBindingBase.transform(self, input, config, **kwargs)
       5553 def transform(
       5554     self,
       5555     input: Iterator[Input],
       5556     config: Optional[RunnableConfig] = None,
       5557     **kwargs: Any,
       5558 ) -> Iterator[Output]:
    -> 5559     yield from self.bound.transform(
       5560         input,
       5561         self._merge_configs(config),
       5562         **{**self.kwargs, **kwargs},
       5563     )


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/runnables/base.py:1424, in Runnable.transform(self, input, config, **kwargs)
       1421             final = ichunk
       1423 if got_first_val:
    -> 1424     yield from self.stream(final, config, **kwargs)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py:398, in BaseChatModel.stream(self, input, config, stop, **kwargs)
        395     self.rate_limiter.acquire(blocking=True)
        397 try:
    --> 398     for chunk in self._stream(messages, stop=stop, **kwargs):
        399         if chunk.message.id is None:
        400             chunk.message.id = f"run-{run_manager.run_id}"


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:1034, in ChatGoogleGenerativeAI._stream(self, messages, stop, run_manager, tools, functions, safety_settings, tool_config, generation_config, cached_content, tool_choice, **kwargs)
       1008 def _stream(
       1009     self,
       1010     messages: List[BaseMessage],
       (...)
       1021     **kwargs: Any,
       1022 ) -> Iterator[ChatGenerationChunk]:
       1023     request = self._prepare_request(
       1024         messages,
       1025         stop=stop,
       (...)
       1032         tool_choice=tool_choice,
       1033     )
    -> 1034     response: GenerateContentResponse = _chat_with_retry(
       1035         request=request,
       1036         generation_method=self.client.stream_generate_content,
       1037         **kwargs,
       1038         metadata=self.default_metadata,
       1039     )
       1041     prev_usage_metadata: UsageMetadata | None = None
       1042     for chunk in response:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:196, in _chat_with_retry(generation_method, **kwargs)
        193     except Exception as e:
        194         raise e
    --> 196 return _chat_with_retry(**kwargs)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/tenacity/__init__.py:336, in BaseRetrying.wraps.<locals>.wrapped_f(*args, **kw)
        334 copy = self.copy()
        335 wrapped_f.statistics = copy.statistics  # type: ignore[attr-defined]
    --> 336 return copy(f, *args, **kw)


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/tenacity/__init__.py:475, in Retrying.__call__(self, fn, *args, **kwargs)
        473 retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        474 while True:
    --> 475     do = self.iter(retry_state=retry_state)
        476     if isinstance(do, DoAttempt):
        477         try:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/tenacity/__init__.py:376, in BaseRetrying.iter(self, retry_state)
        374 result = None
        375 for action in self.iter_state.actions:
    --> 376     result = action(retry_state)
        377 return result


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/tenacity/__init__.py:398, in BaseRetrying._post_retry_check_actions.<locals>.<lambda>(rs)
        396 def _post_retry_check_actions(self, retry_state: "RetryCallState") -> None:
        397     if not (self.iter_state.is_explicit_retry or self.iter_state.retry_run_result):
    --> 398         self._add_action_func(lambda rs: rs.outcome.result())
        399         return
        401     if self.after is not None:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/concurrent/futures/_base.py:449, in Future.result(self, timeout)
        447     raise CancelledError()
        448 elif self._state == FINISHED:
    --> 449     return self.__get_result()
        451 self._condition.wait(timeout)
        453 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/concurrent/futures/_base.py:401, in Future.__get_result(self)
        399 if self._exception:
        400     try:
    --> 401         raise self._exception
        402     finally:
        403         # Break a reference cycle with the exception in self._exception
        404         self = None


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/tenacity/__init__.py:478, in Retrying.__call__(self, fn, *args, **kwargs)
        476 if isinstance(do, DoAttempt):
        477     try:
    --> 478         result = fn(*args, **kwargs)
        479     except BaseException:  # noqa: B902
        480         retry_state.set_exception(sys.exc_info())  # type: ignore[arg-type]


    File /opt/homebrew/Caskroom/miniforge/base/envs/test_langchain/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:190, in _chat_with_retry.<locals>._chat_with_retry(**kwargs)
        187         raise ValueError(error_msg)
        189 except google.api_core.exceptions.InvalidArgument as e:
    --> 190     raise ChatGoogleGenerativeAIError(
        191         f"Invalid argument provided to Gemini: {e}"
        192     ) from e
        193 except Exception as e:
        194     raise e


    ChatGoogleGenerativeAIError: Invalid argument provided to Gemini: 400 * GenerateContentRequest.contents[2].parts[0].function_response.name: Name cannot be empty.

```python

# llama3.1 70B (Together.ai)
execute_agent(
    llama,
    tools,
    "Search AI related news and write it in Instagram post format",
    "llama3.1 70B",
)
```

    [ 'llama3.1 70B' 실행 결과 ]
    "Hey everyone!

    Want to stay up-to-date on the latest AI news?

    Here are some recent articles from Google News:

    1. "딥시크가 '챗GPT' 베낀 증거 있어"..."오픈AI, 합성데이터 활용 방지 예고" - AI타임스
    2. "오픈AI 등 ″딥시크 조사…데이터 무단 수집 여부″" - 연합뉴스TV
    3. "AI로 제조업 안전사고 예방한다…산업부, R&D 사업 공모" - 연합뉴스
    4. "산업부, 제조안전고도화기술개발사업 공고...AI로 제조업 안전사고 예방한다" - AI타임스
    5. "AI로 제조업 산재 위험 예방…산업부, 배터리·석유화학 분야 과제 공모" - 뉴시스

    Check out these articles to learn more about the latest developments in AI!

    #AI #GoogleNews #LatestNews"

```python
# llama3.1 8B (ollama)
execute_agent(ollama, tools, query, "llama3.1(Ollama)")

```

    [ 'llama3.1(Ollama)' 실행 결과 ]
    **AI 투자 뉴스**

    *   **태국 BOI, 5억 달러 규모의 투자 승인** 태국 BOI는 TikTok의 데이터 호스팅 프로젝트를 포함한 총 5억 달러 규모의 투자를 승인했다. 또한 바이오 경제에 대한 혁신적인 인센티브를 발표했다.
    *   **42조 원 규모의 탄약, 'AI' 전부** IT 조선은 42조 원 규모의 탄약이 모두 AI 관련 사업이라고 보도했다.
    *   **트럼프 대통령, 미국 내에서의 AI 관련 사업에 77조 엔 설비 투자 발표** nhk.or.jp는 트럼프 대통령이 미국 내에서의 AI 관련 사업에 77조 엔 규모의 설비 투자를 발표했다고 보도했다.
    *   **투스 시큐리티, 2025년 미국 투자 전망 발표** KIPOST는 투스 시큐리티가 2025년 미국 투자 전망을 발표했다고 보도했다.
    *   **외식업 댓글 관리 AI' 르몽, 블루포인트에서 시드투자 유치** 스타트업엔(StartupN)은 외식업 댓글 관리 AI인 르몽이 블루포인트에서 시드 투자를 유치했다고 보도했다.

```python

# qwen2.5 7B (ollama)
query = "AI 투자와 관련된 뉴스를 검색하고, 결과를 Instagram 게시글 형식으로 작성해 주세요. 한글로 답변하세요!"

execute_agent(qwen, tools, query, "qwen2.5(Ollama)")
```

    [ 'qwen2.5(Ollama)' 실행 결과 ]
    Instagram 게시글:

    🚀 AI 투자 뉴스 🚀

    1. MS, 메타 "AI 투자 지속"에 추가 상승 달렸다 [김현석의 월스트리트나우] - 한국경제

    2. 트럼프발 AI 투자, 한국기업에 기회?… "전력 설루션 시장 주목" - 연합뉴스

    3. ETF·AI 투자의 새 물결… 금리 하락기의 전략적 자산 배분 가이드 [AI PRISM*주간 금융상품 투자자 뉴스] - 서울경제신문

    4. AI 투자 적게 했다는 이유로 애플은 3% 급등, 시총 1위 탈환 - 뉴스1

    5. 어닝서프라이즈 메타 4% 급등.. “AI인프라 투자 계속” - 매일경제

    # 참고 링크:
    1. [MS, 메타 "AI 투자 지속"에 추가 상승 달렸다](https://news.google.com/rss/articles/CBMiWkFVX3lxTE8yeEdfOWxJSmtHSUJYS0ZXVDJaU1RWUTNzU01EVkJQaFpOREZNRE95REpjWU9YR3ZJTlAyODJVdzU2djNZSW5kUjRpVkdid3lfYU9USlNlLTRuZ9IBVEFVX3lxTE1xVzJZZ1lRQ3R6UnlwYkkwS1dsSVBWV2o5TmZoTjBDMGtRanZzMENMMlE2bE9PbHpGMjVsQ0ZOYlk1Wk96bWJGMzlkQ3pqVVZvTndmcw?oc=5)
    2. [트럼프발 AI 투자, 한국기업에 기회?… "전력 설루션 시장 주목"](https://news.google.com/rss/articles/CBMiW0FVX3lxTE4zbUlUMW5PZ2xBWVBXQjRrdWM1ZlRlN3BoWE1VRkd3cHRlcTVKSEEtLUZrMWlKc2NkRFY5dmJ3YmlmUTctd2Q4SzNxQnJBZnJMYzJxOVo1WjB1em_SAV5BVV95cUxQWjVzN2t0ejY2akhjZGVGbnpRRWw3ZWUwckFLc3U3NXkwaFcxaTVCRWJHY0M0STNhMU1heURwWmlXa285YnJqa2h5NFdkamVEejhwQnpVNE9uVS1aWk9B?oc=5)
    3. [ETF·AI 투자의 새 물결… 금리 하락기의 전략적 자산 배분 가이드](https://news.google.com/rss/articles/CBMiVkFVX3lxTFA4WnRkSE1kZnUteWhMdmJWSWFEWUs3Y0tTTVFjc0M2YmNNOEM5b0RHa3pDN0dneXp6ZG9tcGNiQjMtYXF0X2l6a3VoMWlrTHdSVDgyLXlR?oc=5)
    4. [AI 투자 적게 했다는 이유로 애플은 3% 급등, 시총 1위 탈환](https://news.google.com/rss/articles/CBMiWEFVX3lxTE5IYk80R1pzV084dlpXODE3RkFOc2pVNzdDMkxFUzFGeE51QUZrb2tMMVo5cUptTExXS01jZEFlTXBQUWlTRXBmYUhfUjNXc24xU1daQkE?oc=5)
    5. [어닝서프라이즈 메타 4% 급등.. “AI인프라 투자 계속”](https://news.google.com/rss/articles/CBMiUkFVX3lxTE5kVkZWUWxKZXRuUzQxRkFOc2pVNzdDMkxFUzFGeE51QUZrb2tMMVo5cUptTExXS01jZEFlTXBQUWlTRXBmYUhfUjNXc24xU1daQkE?oc=5)

    # #AI #투자 #주식 #증권 #경제 #시황 #마켓뉴스

    # (이 게시글은 참고용으로 작성되었으며, 실제 투자는 주의가 필요합니다. 전문적인 자문을 받으시기 바랍니다.)
