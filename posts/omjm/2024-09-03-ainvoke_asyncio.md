---
title: LangChain의 ainvoke와 asyncio를 이용한 LLM 병렬처리 최적화
---

## ainvoke와 asyncio

> LLM(대규모 언어 모델)을 활용한 응용 프로그램을 개발할 때, 성능 최적화는 중요한 과제입니다. 특히 여러 요청을 동시에 처리해야 하는 경우, 비동기 프로그래밍이 유용합니다. 이번 포스팅에서는 Python의 asyncio와 LangChain의 비동기 기능을 활용하여 LLM 기반 API를 최적화하는 방법을 설명하겠습니다. 이 방법은 다양한 응용 프로그램에서 처리 속도를 크게 향상시킬 수 있습니다.

### 프로젝트 개요

식당의 메뉴 데이터를 크롤링하고, 크롤링된 데이터를 데이터베이스에 저장한 후, 저장된 메뉴의 영양성분을 추출하는 API를 개발했습니다. 이 API는 특히 LLM을 호출하는 부분에서 성능 최적화가 필요했기 때문에, asyncio와 LangChain을 활용해 이를 해결했습니다.

### 1. 비동기 LLM 호출의 필요성

기본적으로 LLM을 호출할 때마다 네트워크 요청이 발생하며, 이 과정에서 지연이 발생할 수 있습니다. 특히 여러 요청을 순차적으로 처리할 경우, 전체 응답 시간이 길어질 수 있습니다. 이를 해결하기 위해, 여러 요청을 비동기적으로 동시에 처리하면 전체 처리 시간을 줄일 수 있습니다.

### 2. 비동기 LLM 호출의 구현

비동기 LLM 호출을 구현하는 첫 단계는 비동기 함수를 작성하는 것입니다. 예를 들어, 각 음식 메뉴에 대해 영양성분을 추출하는 작업을 비동기 함수로 작성할 수 있습니다. 이 함수는 LLM에 요청을 보내고, 그 결과를 **ainvoke**를 통해 비동기적으로 반환합니다.

```python

from langchain_core.prompts import PromptTemplate
from typing import List, Dict
from db_set import settings
from langchain_openai import ChatOpenAI


# 비동기 LLM 호출 함수 예시
async def extract_nutrient_info_async(products: Dict) ->  Dict:
	# llm 초기화
    llm = ChatOpenAI(model='gpt-4o-mini', api_key=settings.OPENAI_API_KEY, temperature=0)

    prompt = f"""
    제품 이름: {product_name} (ID: {product_id}).
    이 제품은 한국에서 판매하고 있는 일식 중 하나 입니다.
    일본발음을 그대로 변경한 메뉴이름들인것을 참고하여 들어간 재료와 1인분 양의 영양성분을 추정하고,
    그 이유를 설명하세요 :
    - 탄수화물 (g)
    - 당 (g)
    - 단백질 (g)
    - 지방 (g)
    - 포화지방 (g)
    - 불포화지방 (g)
    - 콜레스테롤 (mg)
    - 섬유질 (g)
    - 나트륨 (mg)
    - 칼륨 (mg)
    - 칼로리 (kcal)
    """

     llm_chain = prompt | llm

    prompt_inputs = {
        'product_id': products.get('id'),
        'product_name': products.get('name')
    }
    response = llm_chain.ainvoke(prompt_inputs)

    return response
```

extract_nutrient_info_async 함수는 각 메뉴의 이름과 ID를 LLM에 보내어 영양성분을 추정합니다. ainvoke 메서드를 사용하여 비동기 방식으로 LLM에 요청을 보냅니다.

### 2-1 ainvoke 함수:

> ainvoke는 LangChain에서 비동기 작업을 처리하기 위해 제공되는 메서드로, LLM을 사용할 때 비동기적으로 요청을 보내고 결과를 반환받을 수 있습니다. 이 방법은 특히 여러 개의 요청을 동시에 처리해야 하거나 LLM 호출이 시간이 오래 걸릴 때 유용합니다. 비동기 호출은 Python의 asyncio 라이브러리를 활용해 동작하며, 네트워크 대기 시간을 줄이고 시스템 리소스를 효율적으로 사용할 수 있습니다.

### 3. 최종 파이프라인 구축

마지막으로, 크롤링된 데이터를 저장하고, 영양성분을 추출하는 전체 파이프라인을 구축했습니다. 이 파이프라인은 FastAPI를 통해 API로 구현되었으며, 비동기 처리 부분을 통해 성능이 최적화되었습니다.

```python
import os
import sys

#현재보다 상위 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
batch_dir = os.path.dirname(current_dir)
sys.path.append(batch_dir)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import CrawlRequest
from models import Brand
from mysql_db import get_db
from restaurants import elt_db, b_inference, a_diningcode_crawling
import asyncio

router = APIRouter(
    prefix= "/api",
    tags=["product"]
)


# 키워드 저장 파이프라인 구축
@router.post("/keywords_pipeline/")
def save_product(data: CrawlRequest, db: Session = Depends(get_db)):
    try:
        # 1. 크롤링 단계
        menu_info = a_diningcode_crawling.get_menu_info(data.restaurant_id)

        # menu_info가 None인지 확인
        if menu_info is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve menu information")

        # 2. 크롤링된 데이터를 DB에 저장
        elt_db.save_product_info(db, menu_info, data.center_keyword_name)

        # 3. 저장된 brand_id를 가져옴
        brand = db.query(Brand).filter(Brand.name == menu_info["restaurant_name"]).first()
        if not brand :
            raise HTTPException(status_code=404, detail = "Brand not found")

        # 4. 제품 이름 목록 불러오기
        products = elt_db.get_product_names(db, brand.id)

        # 각 제품에 대해 영양성분 리스트 초기화
        nutrient_data_list = []

        # 비동기 작업을 위해 asyncio 이벤트 루프 시작
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tasks = []

        # 5. 각 제품에 대해 영양성분 추출 및 저장
        for product in products:
            product_info = {
                "id": product["product_id"],  # 제품 ID
                "name": product["product_name"]  # 제품 이름
            }
             # 비동기 함수 호출 준비
            tasks.append(b_inference.extract_nutrient_info_async(product_info))

        # 비동기 작업 실행
        nutrient_results = loop.run_until_complete(asyncio.gather(*tasks))


        for product, nutrient_data in zip(products, nutrient_results):
            nutrient_data_list.append({
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'nutrient_data': nutrient_data
            })

        return {
            "message": "Nutrient data extracted successfully",
            "data": nutrient_data_list
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")  # 에러 내용을 로그로 남깁니다.
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


```

### 3-1 asyncio

> asyncio는 Python에서 비동기 프로그래밍을 지원하는 표준 라이브러리입니다. 이 라이브러리를 통해 개발자는 단일 스레드에서 여러 작업을 효율적으로 동시에 처리할 수 있습니다. asyncio는 비동기 I/O 작업을 처리하기 위해 이벤트 루프(event loop)를 사용하며, 이는 주어진 작업이 완료될 때까지 다른 작업을 블로킹하지 않고 계속해서 실행할 수 있게 합니다.

#### 비동기 작업을 위해 asyncio 이벤트 루프 시작

```python
tasks = []

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
```

- loop = asyncio.new_event_loop():
  이 코드 줄은 새로운 이벤트 루프를 생성합니다. 이벤트 루프는 비동기 작업을 관리하고 실행하는 중요한 역할을 합니다. 비동기 작업은 일반적인 함수 호출과는 다르게 동시에 여러 작업을 실행할 수 있습니다. 이 이벤트 루프는 그 작업들이 어떻게 실행될지를 관리합니다.
- asyncio.set_event_loop(loop):
  새로 생성된 이벤트 루프를 현재 이벤트 루프로 설정합니다. Python은 기본적으로 실행 중인 이벤트 루프를 관리하는 기능을 제공합니다. 이 코드 줄은 우리가 새로 만든 이벤트 루프를 현재 실행 중인 루프로 설정하여, 이후 비동기 작업들이 이 루프에서 실행되도록 합니다.
- tasks = []:
  • tasks는 리스트로, 비동기 작업들을 저장할 공간입니다. 이 리스트에 저장된 각 작업은 이후에 한꺼번에 실행됩니다. 이 접근법은 여러 개의 비동기 작업을 병렬로 실행할 수 있도록 도와줍니다.

### 4. 결과

![](https://velog.velcdn.com/images/looa0807/post/7d023357-0d30-4c41-bddb-db631c5129e4/image.png)

이 여러개의 호출이 거의 동시에 실행이되는것을 볼 수 있습니다.

![](https://velog.velcdn.com/images/looa0807/post/6b5ff257-5b82-45e6-9ff4-7acc44b51e36/image.png)
프롬프트결과가 요청한 메뉴이름, 아이디와 함께 잘 나오는 것을 확인 할 수 있습니다. 현재는 프롬프트의 응답을 텍스트형식으로 받았습니다. 앞으로 nutrient를 json스타일로 변경하여 영양성분을 데이터베이스에 적재하고 키워드를 뽑는 일이 남았다고 할 수 있겠습니다.

### 5. 성능 향상 결과

비동기 처리를 도입한 결과, 여러 제품에 대한 영양성분 추출 작업이 크게 빨라졌습니다. 이 방법을 LLM 시스템의 응답 시간을 줄이고, 동시에 처리할 수 있는 요청의 수를 크게 늘릴 수 있었습니다.

> 이번 글에서는 asyncio와 LangChain의 비동기 기능을 활용하여 LLM 기반 API의 성능을 최적화하는 방법을 다뤘습니다. 비동기 처리를 통해 여러 작업을 동시에 처리함으로써 처리 속도를 크게 향상시킬 수 있었습니다. 앞으로도 이러한 비동기 방식을 활용해 더 효율적인 LLM 기반 응용 프로그램을 개발할 수 있을 것입니다.
