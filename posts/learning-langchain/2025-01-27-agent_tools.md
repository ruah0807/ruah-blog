---
title: "LangChain 도구(Tools)"
---

# 도구 (Tools)

도구(Tool)는 에이전트, 체인 또는 LLM이 외부 세계와 상호작용하기 위한 인터페이스입니다.

LangChain 에서 기본 제공하는 도구를 사용하여 쉽게 도구를 활용할 수 있으며, 사용자 정의 도구(Custom Tool) 를 쉽게 구축하는 것도 가능합니다.

LangChain 에 통합된 도구 리스트는 아래 링크에서 확인할 수 있습니다.

https://python.langchain.com/v0.1/docs/integrations/tools/

```python
from dotenv import load_dotenv


load_dotenv()


```

    True

```python
# Langsmith 추적설정
from langchain_teddynote import logging
logging.langsmith("agent-tools")
```

    LangSmith 추적을 시작합니다.
    [프로젝트명]
    agent-tools

```python
import warnings

#경고 메시지 무시
warnings.filterwarnings("ignore")

```

## 빌트인 도구(built-in tools)

랭체인에서 제공하는 사전에 정의된 도구(tool) 와 툴킷(toolkit) 을 사용할 수 있다.
tool 은 단일 도구를 의미하며, toolkit 은 여러 도구를 묶어서 하나의 도구로 사용할 수 있다.
관련 도구는 아래의 링크에서 참고 가능하다.

- Resource : https://python.langchain.com/docs/integrations/tools/

### Python REPL 도구

이 도구는 Python 코드를 REPL(Read-Eval-Print Loop) 환경에서 실행하기 위한 클래스를 제공한다.

- 코드를 만들어 실행하는 도구
- Python 셸 환경을 제공
- 유효한 Python 명령어를 입력으로 받아 실행
- 결과를 보려면 print(...) 함수를 사용

- 주요 특징

  - sanitize_input: 입력을 정제하는 옵션 (기본값: True)
  - python_repl: PythonREPL 인스턴스 (기본값: 전역 범위에서 실행)

- 사용 방법
  - PythonREPLTool 인스턴스 생성
  - run 또는 arun, invoke 메서드를 사용하여 Python 코드 실행
- 입력 정제
  - 입력 문자열에서 불필요한 공백, 백틱, 'python' 키워드 등을 제거

```python
! pip install -q langchain_experimental
```

```python
from langchain_experimental.tools import PythonREPLTool

# 파이썬 코드 실행도구 생성
python_tool = PythonREPLTool()

```

```python
# 파이썬 코드 실행 및 반환
print(python_tool.invoke("print(200+225)"))
```

    425

아래는 LLM에게 파이썬 코드를 작성하도록 요청하고 결과를 반환하는 예제이다.

#### Flow

    1. LLM에게 파이썬 코드를 작성하도록 요청
    2. 파이썬 코드를 작성하고 실행
    3. 결과를 반환

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# execute python, print the process, and return the result
def print_and_execute(code, debug = True):
    if debug:
        print(f"Executing code:")
        print(code)
    return python_tool.invoke(code)


# Request the code to make a python code
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 레이몬드 헤팅이며, 메타 프로그래밍과 우아하고 간결하며 짧지만 잘 문서화된 코드를 잘 아는 전문가 파이썬 프로그래머입니다.
            당신은 PEP8 스타일 가이드를 따릅니다.
            Return only the code, no intro, no chatty, no markdown, no code block, no nothing. Just the code.
            """
        ),
        (
            "user",
            "{input}"
         )

    ]
)

# create llm model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# create chain using prompt, llm, and output parser
# RunnableLambda : 함수를 래핑하여 함수를 실행하는 데 사용되는 클래스
chain = prompt | llm | StrOutputParser() | RunnableLambda(print_and_execute)
```

이 때, 왜인지 모르겠지만 gpt-4o 이상의 모델을 실행해야만 runnableLambda에서 함수 결과도 출력된다.

```python
# print the result
print(chain.invoke("로또 자동 번호 생성기를 출력하는 코드를 작성하시오"))
```

    Executing code:
    import random

    def generate_lotto_numbers():
        return sorted(random.sample(range(1, 46), 6))

    print(generate_lotto_numbers())
    [8, 26, 30, 35, 37, 43]

### 검색 API 도구

최신 날씨, 뉴스 등의 경우는 LLM이 알수없기 때문에 인터넷 검색을 해야한다.

- Search API 도구 : DuckDuckGo Search, SERP, Tavily

우리는 그중에서 Tavily 도구를 사용해 보겠습니다. 먼저 API 키를 발급받아야 한다.
Tavily는 한달에 1000건까지의 request를 무료로 사용할 수 있다고 한다.

- https://app.tavily.com/home

발급받은 키를 .env 파일에 저장한다.

```terminal
TAVILY_API_KEY=tvly-***************************
```

#### TabilySearchResult

- Tavily 검색 API를 쿼리하고 Json 형태로 반환하는 도구
- 포괄적이고 정확하며 최신 정보를 제공
- 현재 이벤트, 주요 뉴스, 주요 블로그 등을 포함한 질문에 답변할때 유용

- 주요 매개변수

  - `max_results`(int): 검색 결과 최대 개수 (기본값: 5)
  - `search_depth`(str): 검색 깊이 (basic, advanced)
    - basic : 기본 검색 결과를 반환, 무료
    - advanced : 더 깊은 검색 결과를 반환하지만 유료
  - `include_domains`(list[str]): 검색 결과에 포함할 도메인 목록
  - `exclude_domains`(list[str]): 검색 결과에 포함하지 않을 도메인 목록
  - `include_answer`(bool): 검색 결과에 대한 짧은 답변 포함 여부
  - `include_raw_content`(bool): 원본 콘텐츠 포함 여부
  - `include_images`(bool): 이미지 포함 여부

- 반환값
  - 검색 결과를 포함하는 Json 형식의 문자열(url, content)

```python
import os
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# create tool
tool = TavilySearchResults(
    max_results=10,
    include_answer=True,
    include_raw_content=True,
    include_domains = ["google.com", "github.com"],
    # include_images = True,
    # include_depth = "basic",
    # exclude_domains = []
)

# invoke tool
tool.invoke({"query" : "langchain으로 tavily를 사용하는방법"}) # 'query' 키는 필수 매개변수이며, **':'(세미콜론)** 뒤에 쿼리 문자열을 작성한다.

```

    [{'url': 'https://github.com/langchain-ai/langchain/issues/29423',
      'content': "Checked other resources. I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and didn't find it."},
     {'url': 'https://github.com/pakagronglb/simple-langchain-translator',
      'content': "A simple yet powerful translation API built with LangChain and FastAPI. This project demonstrates how to create a translation service using OpenAI's language models through LangChain's inte"},
     {'url': 'https://cloud.google.com/spanner/docs/langchain',
      'content': 'Preview — LangChain This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms.Pre-GA features are available "as is" and might have limited support.'},
     {'url': 'https://github.com/langchain-ai/weblangchain/blob/main/main.py',
      'content': 'Saved searches Use saved searches to filter your results more quickly'},
     {'url': 'https://cloud.google.com/docs/generative-ai/code-samples?hl=ko',
      'content': '금융 상담사에게 투자 추천을 제공하는 Node 기반 RAG 앱을 빌드하는 방법을 알아봅니다. 이 샘플은 Vertex AI, Cloud Run, AlloyDB, Cloud Run 함수와 통합됩니다. Angular, TypeScript, Express.js, LangChain으로 빌드되었습니다.'},
     {'url': 'https://developers.google.com/optimization/',
      'content': "OR-Tools \xa0|\xa0 Google for Developers OR-Tools ========= OR-Tools OR API Installation Guides Reference Examples Support OR-Tools OR-Tools OR-Tools Get started with OR-Tools Install OR-Tools Install OR-Tools About OR-Tools OR-Tools is an open source software suite for optimization, tuned for tackling the world's toughest problems in vehicle routing, flows, integer and linear programming, and constraint programming. After modeling your problem in the programming language of your choice, you can use any of a half dozen solvers to solve it: commercial solvers such as Gurobi or CPLEX, or open-source solvers such as SCIP, GLPK, or Google's GLOP and award-winning CP-SAT. Join Google OR-Tools Discord server Google API Console Google Cloud Platform Console Google Play Console Actions on Google Console Google Home Developer Console"},
     {'url': 'https://chromewebstore.google.com/detail/simple-translator-diction/lojpdfjjionbhgplcangflkalmiadhfi',
      'content': 'Accurate translate words, phrases and texts using Google Translate. Right click on the word or phrase to translate using the Google Translator. Right-click on any word or phrase to access Google Translator for instant online translation. Quickly translate words on the current web page using Google Translator by right-clicking on the desired text. Use the "Web Translator" extension at your own risk; no warranty is provided. Use the translation button in the context menu or the extension\'s popup. The "Web Translator" Chrome extension is independent of the popular site https://translate.google.com. Google Translate By the Google Translate team. Accurate translate words, phrases and texts using Google Translate. Right-click to translate words or phrases with Google Translate: accurate translations, full-page translator'},
     {'url': 'https://www.google.com/intl/ta/inputtools/',
      'content': 'Google உள்ளீட்டு கருவி உள்ளீட்டு கருவி உள்ளடக்கத்திற்குச் செல் முகப்பு இதை முயற்சிக்கவும் Chrome Google சேவைகள் உங்கள் சொற்கள், உங்கள் மொழி, எங்கும் Google சேவைகள், Chrome, Android சாதனங்கள் மற்றும் Windows ஆகியவற்றிற்காக கிடைக்கின்றது. இதை முயற்சிக்கவும் வீடியோவைக் காண்க ஆன்லைன், ஆஃப்லைன், பயணத்தின்போது வீட்டில், பணியில் அல்லது வேறு எங்காவது இருக்கும்போது—தேவைப்படும்போது, வேண்டிய மொழியில் தொடர்புகொள்ளலாம். உங்களுக்காக நீங்களே தனிப்பயனாக்கியது Google Input Tools உங்கள் திருத்தங்களை நினைவுபடுத்துகிறது மற்றும் புதிய அல்லது பொதுவில் இல்லாத சொற்கள் மற்றும் பெயர்களுக்கான தனிப்பயன் அகராதியைப் பராமரிக்கிறது. நீங்கள் விரும்பும் வழியில் தட்டச்சு செய்க உங்கள் செய்தியை எல்லா மொழிகளிலும், நீங்கள் விரும்பும் நடையிலும் பெறுக. 80 க்கும் மேற்பட்ட மொழிகளுக்கு இடையில் மாற முடியும், மேலும் உள்ளீட்டு முறைகளானது தட்டச்சு செய்வது போலவே எளிதானது. பிற மொழிகளில் உள்ள உள்ளீட்டு முறைகள்: 日本語入力 ஆதரிக்கப்படும் மொழிகள் உள்ளடக்கப் பண்புக்கூறு நீங்கள் நினைப்பதைத் தெரியப்படுத்துங்கள் – கருத்தைச் சமர்ப்பி. மொழியை மாற்று:  Google Google ஓர் அறிமுகம் தனியுரிமை விதிமுறைகள்'},
     {'url': 'https://www.google.com/inputtools/try/',
      'content': 'Try Google Input Tools online – Google Input Tools Input Tools Skip to content On Google Services Try Google Input Tools online Google Input Tools makes it easy to type in the language you choose, anywhere on the web. To try it out, choose your language and input tool below and begin typing. Special Characters Get Google Input Tools Content attribution Google About Google Latin South Asian Scripts Southeast Asian Scripts Other East Asian Scripts Han 1-Stroke Radicals Han 2-Stroke Radicals Han 3-Stroke Radicals Han 4-Stroke Radicals Han 5-Stroke Radicals Han 6-Stroke Radicals Han 7-Stroke Radicals Han 8-Stroke Radicals Han 9-Stroke Radicals Han 10-Stroke Radicals Han 11..17-Stroke Radicals Han - Other Latin 1 Supplement Math Math Alphanumeric Insert special characters'},
     {'url': 'https://chromewebstore.google.com/detail/google-translate/aapbdbdomjkkjkaonfhkkikfgjllcleb/RK=2/RS=BBFW_pnWkPY0xPMYsAZI5xOgQEE-',
      'content': 'Google Translate - Chrome Web Store Google Translate By the Google Translate team. Learn more about Google Translate at https://support.google.com/translate. UPDATE (v.2.14+): Page translation is now supported natively in Chrome browsers and is no longer supported in the Google Translate extension. google-translate-chrome-extension-owners@google.comPhone Google Translate handles the following: Translator Right click to translate websites and PDF documents using Google Translate Translator Google Translate Plus Translate the definitions by google translate. Translate selected text with Google Translate Accurate translate words, phrases and texts using Google Translate. Translate text on any webpage instantly using Google Translate. Translator Right click to translate websites and PDF documents using Google Translate Translator Google Translate Plus Translate the definitions by google translate. Translate selected text with Google Translate'}]

### 이미지 생성 Tool ( DALL-E )

`DallEAPIWrapper` 클래스 OpenAI의 DALL-E 이미지 생성기를 위한 래퍼(wrapper)
이 도구를 사용하면 DALL-E API를 쉽게 통합하여 텍스트 기반 이미지 생성 기능이 구현 겨ㅏ능하다. 다양한 설정 옵션을 통해 유연하고 강력한 이미지 생성 도구로 활용할 수 있다.

- Element
  - `model`: 사용할 DALL-E 모델 이름 (기본값: "dall-e-2", "dall-e-3")
  - `n`: 생성할 이미지 수 (기본값: 1)
  - `size`: 생성할 이미지 크기
    - "dall-e-2": "1024x1024", "512x512", "256x256"
    - "dall-e-3": "1024x1024", "1792x1024", "1024x1792"
  - `style`: 생성될 이미지의 스타일 (기본값: "natural", "vivid")
  - `quality`: 생성될 이미지의 품질 (기본값: "standard", "hd")
  - `max_retries`: 생성 시 최대 재시도 횟수
- Function
  - DALL-E API를 사용하여 텍스트 설명에 기반한 이미지 생성
- Flow
  1. LLM에게 이미지를 생성하는 프롬프트를 작성하도록 요청
  2. DALL-E API를 사용하여 이미지를 생성
  3. 생성된 이미지를 반환

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# initialize
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=1000)

# Define the prompt for the DALL-E image generation
prompt = PromptTemplate.from_template(
    "Generate a detailed IMAGE GENERATION prompt for DALL-E based on the following description"
    "Return only the prompt, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the prompt."
    "Output should be less than 1000 characters. Write in English only."
    "Image Description : \n{image_desc}"
)

# create chain
chain = prompt | llm | StrOutputParser()

# invoke chain
image_prompt = chain.invoke(
    {"image_desc" : "A theater with a stage which is performing the musical of 'Harry Potter' and a large audience"}
)

print(image_prompt)

```

    A grand theater interior filled with an enthusiastic audience, all eyes directed towards a vibrant stage showcasing the musical production of 'Harry Potter.' The stage is adorned with whimsical, magical elements inspired by the Harry Potter universe, such as floating candles, a backdrop of Hogwarts Castle, and colorful banners representing the four houses: Gryffindor, Slytherin, Hufflepuff, and Ravenclaw. The actors, dressed in enchanting costumes, are performing an exciting scene filled with spells and magical creatures. The audience, a diverse group of people of various ages, expresses a mix of awe and excitement, some holding wands and wearing themed attire. The theater is illuminated with warm lights, adding to the magical atmosphere, while curtains drape elegantly on the sides. The scene captures the joy and energy of live performance, evoking the wonder of the Harry Potter world.

이제 이미지를 생성해보자.

```python
# DallEAPIWrapper 버그로 인한 임시 버전 다운그레이드 명령어 (실행 후 restart)
# ! pip uninstall langchain==0.2.16 langchain-community==0.2.16 langchain-text-splitters==0.2.4 langchain-experimental==0.0.65 langchain-openai==0.1.20
! pip install langchain langchain-community langchain-text-splitters langchain-experimental langchain-openai
```

```python
# Bring the DALL-E API Wrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from IPython.display import Image
import os

# initialize the DALL-E API Wrapper
dalle = DallEAPIWrapper(
    model="dall-e-3",   # model: 사용할 DALL-E 모델 버전
    size="1024x1024",   # size: 생성할 이미지 크기
    quality="standard",  # quality: 생성할 이미지 품질
    n=1,  # n: 생성할 이미지 수
)

# query
query = "A theater with a stage which is performing the musical of 'Harry Potter' and a large audience"

# create image and get the url
image_url = dalle.run(chain.invoke({"image_desc" : query}))

# display the image
Image(url=image_url)
```

<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-GdjIuIPxJK99DMMVRPzqCmQ9/user-Ay3dqUtJOMGOQXx1DS8wjjUU/img-mkMW3CYHptWVlvM3AnyHbwry.png?st=2025-01-27T12%3A59%3A36Z&se=2025-01-27T14%3A59%3A36Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-27T01%3A11%3A35Z&ske=2025-01-28T01%3A11%3A35Z&sks=b&skv=2024-08-04&sig=MzSWd0s3d1pC0MWp8hKm//yblpQaMeiJvvwcDTnm1XI%3D"/>

## 사용자 정의 도구(Custom Tool)

Langchain에서 제공하는 도구를 사용하는 것 외에도 사용자 정의 도구를 만들어 사용할 수 있다.
이를 위해 `langchain.tools` 모듈의 `tool` 데코레이터를 사용하여 함수를 도구로 변환.

### @tool Decorator

`@tool` 데코레이터는 함수를 도구로 변환하는 기능을 갖고있다. 다양한 옵션을 이용하여 도구의 동작을 정의할 수 있다.
이는 일반 Python 함수를 도구로 쉽게 변환이 가능하여, 자동화된 문서와 유연한 인터페이스 생성이 가능하다.

- instructions
  1. 함수 위에 `@tool` 데코레이터 적용
  2. 필요에 따라 데코레이터 매개변수 설정

```python
from langchain.tools import tool

# change function to tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
```

```python
add_numbers.invoke({"a": 1, "b": 2})
```

    3

```python
multiply_numbers.invoke({"a": 1, "b": 2})
```

    2

```python
! pip install -qU langchain-teddynote
```

```python
from langchain_teddynote.tools import GoogleNews

# create tool
news_tool = GoogleNews()
```

```python
# Search news
news_tool.search_latest(k=3)
```

    [{'url': 'https://news.google.com/rss/articles/CBMiakFVX3lxTE04V2xZMFl1X3ZqT1dGM2ZiUUxEN0NhRmdJeG51bDNXYm1PWl9VdU9CTFZpUlpWbzdvdHZxbWpXM0RNOEpoTXJ2NWIzLUhkaE5hMkdjMDFSMDNOQzJ1QzJCaWJodnBjcXRGMWc?oc=5',
      'content': '권영세 “‘공수처 굴종’ 검찰총장 사퇴하라”…국힘, 검찰 일제히 비판 - 한겨레'},
     {'url': 'https://news.google.com/rss/articles/CBMilwFBVV95cUxNVGpnRFFNTUNmWHNZWFBwcXBsMG8zRy1DSVFqNDRzazF0U0Y1UFZoOWFldkhiZXlfdGEzVzN3RmJZLTV3dGZKUV9BTWhvLS0wOEhqMjZQb2VCMTdhcDFhcktkTkFlQ2pVd25Ha3VGWFh1TFdxTWVqd2RCelBvOENkbkhhcHdLb0psbUx2bl9RQTdNZVZTeHVz0gGrAUFVX3lxTE1lOWxDNmdtVnFieDdzdWtabzZqM0xVUktTTksyQmVCa3gyM1ZoUVZCdGFZRHdhNVZVTE5IUk9rWUJxOXdvUHlOVG15RW9kbmRYQkl6dm56LWdlTUs5SWhndFAyVk4zN09kcU5ZRF9WTVcza3gzR1JObmQ1Zi1KckZuUFpUenRGNFQtU08tVmJKLTVFY3IwNzBuS0lnMk1MdHFHbXI3SGNGXzVBZw?oc=5',
      'content': '폭설 없던 설 연휴…이례적 대설 내리는 이유는 - 조선일보'},
     {'url': 'https://news.google.com/rss/articles/CBMiWkFVX3lxTFBlU1dkT3ZiT3hWSkdMX09SWnE3WVdsWC0wdS00THZXR0F3Q19PNk9qaFpmN2pNU3lKS0J1cjJkYUJwaUpsMG1ZcGlVNUMwMkY0UG1FbnYzWFphd9IBVEFVX3lxTE1qaV91Slh1WDdBU2dfX2NEYUJhSDFBLXpndTBwbkJSNnptamtodnNqOVZjeUxHU0wyMWgxb2llTkd0R08xbm9PYVNDWTBCMzdocHBmZA?oc=5',
      'content': '"부정선거 의혹, 음지에서 양지로"?…심상찮은 분위기 [이슈+] - 한국경제'}]

```python
# Search news using keyword
news_tool.search_by_keyword("AI 투자", k=3)

```

    [{'url': 'https://news.google.com/rss/articles/CBMiU0FVX3lxTE54bzhUUm5kMmNRZE1Rcmplb2s4Ukl0aFNuRjQ4d1JlbnE2NW1ibVc2ZEhxMVJtR2pERjNfajNwYm1pU2JhR1FrN1pSeUdfRmowUVF3?oc=5',
      'content': '트럼프 눈 밖에 나지 않기 위해?…빅테크, 천문학적 AI 투자 발표 - 네이트 뉴스'},
     {'url': 'https://news.google.com/rss/articles/CBMigAFBVV95cUxPTmtqS0xaS1ZoY21kQkU2UktLbnFSakYydE9tRG5tbFZEUWZWaDlOdkJJenJkaTdkRkZOWEg3SmdTSS1CRU1pNHN6ZTFNQ3B4S3BJR0RaR3RZQ2puU01qd042Mm1ROWdxRVh6Y3RxeHpIb1VrR1lUUTVIWTBuTmlRSQ?oc=5',
      'content': '[美증시프리뷰]제프리즈 "딥시크 여파로 AI 설비 투자 감소할수도" - 이데일리'},
     {'url': 'https://news.google.com/rss/articles/CBMiiAFBVV95cUxNbFhCblQyUnF1NXF5Q2FQRHBmdGZGS3ZUSFBTMUdZZ2tyQUU2VHBfWHRqYXJ3TEdkOUc1VHI4YnFPUkpUZncyT2xEZ1ZpX29KV2hIa2hCcFFwUGRVQlB3YzVlVjJOT0wydjVvTENoM0N4cnppaFF6UFpiNmJ2RWhGOUFPVERpRUZS?oc=5',
      'content': "AI 투자 붐, 글로벌 ‘신디케이트론’ 시장 견인...2024년 6조 달러 돌파 '사상 최대' - 글로벌이코노믹"}]

```python
from langchain_teddynote.tools import GoogleNews
from langchain_core.tools import tool
from typing import List, Dict

# create tool as a searching news with keyword
@tool
def search_keyword_news(query: str) -> List[Dict[str,str]]:
    """Search news with keyword"""
    print(query)
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=3)

# invoke tool
# print(search_keyword_news.invoke("AI 투자"))
```

```python
# result
search_keyword_news.invoke({"query": "한국 정치 현황"})
```

    한국 정치 현황





    [{'url': 'https://news.google.com/rss/articles/CBMiWkFVX3lxTE5nOG43RElxREdRcFYtenpDeHR4ZVNGR185YmhnYVg2TkUyMXRONG80a0NRNWRKVWVrUGRCV2ZxdnJUaWlXa0VSQ0t6TU93ZVBIdUlvbGppV3puZ9IBVkFVX3lxTE0tQjR4UHZWQnp4cXFUOGpTY3dBTVo0b1ZZQkZwUnhWTkJmNm9vOFdPN2pLdzA5YTlEZkVrNTEtYk1WWkVLODIyN0wxVGNXdEgxejF1eVlB?oc=5',
      'content': '나경원 트럼프 취임식서 한국 정치현황 알릴 것 - 아주경제'},
     {'url': 'https://news.google.com/rss/articles/CBMiU0FVX3lxTE1aTHl2b0RvcWQyVGpyTTdsbEJuc1RLQzltNFBzYXhsRGJDR1NCYUNkeW1FejlTZGVlSWg3VUlQZVZyTnNWejlSWUViZmpxSmhTQWgw?oc=5',
      'content': '김동연, 미디어리더들과 대화…한국 정치·경제 통찰 공유 - 네이트 뉴스'},
     {'url': 'https://news.google.com/rss/articles/CBMiXEFVX3lxTE9Ddmo2bW1wcW1FaVNKSjRRdVNnUGduOS1RN2h0Qlg4WWZlLVlJRzVOTXBqWTBQdXR6MDlqSXRNTWJLUDRzU25uMVYySzVJMVN6Mld6VmFmaVFydzNy0gFTQVVfeXFMUDZFYXBvUjF2UlcxNHJRVnByaHBDazg1QS1zZFRSSk00aGxzcFJOZlJwcGJweHJDaE45UEVYMHh2Z0hXOTlSQXk5SXA0aFhFVThhemM?oc=5',
      'content': '다보스서 엘 고어 만난 김동연, 계엄 이후 한국 정치·경제 회복 탄력성 강조 - KPI뉴스'}]

```
<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-GdjIuIPxJK99DMMVRPzqCmQ9/user-Ay3dqUtJOMGOQXx1DS8wjjUU/img-mkMW3CYHptWVlvM3AnyHbwry.png?st=2025-01-27T12%3A59%3A36Z&se=2025-01-27T14%3A59%3A36Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-27T01%3A11%3A35Z&ske=2025-01-28T01%3A11%3A35Z&sks=b&skv=2024-08-04&sig=MzSWd0s3d1pC0MWp8hKm//yblpQaMeiJvvwcDTnm1XI%3D"/>
```
