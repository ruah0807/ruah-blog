const e=`> 이번 프로젝트에서는 RAG(검색 증강 생성) 기반의 레시피 키워드 도출 시스템을 구축하기 위해 LangGraph를 활용했습니다. LangGraph는 복잡한 워크플로우를 직관적이고 확장 가능한 방식으로 설계하고 시각화할 수 있는 도구로, 특히 조건부 엣지 기능을 통해 유연한 로직 처리가 가능합니다. 이 블로그에서는 프로젝트의 주요 구현 사항과 LangGraph의 활용 방법을 중심으로 설명하겠습니다.

### LangGraph

LangGraph는 LangChain의 구성 요소 중 하나로, 워크플로우를 노드 기반으로 설계하고 시각화할 수 있게 해주는 도구입니다. 각 노드는 독립적인 작업을 수행하며, 노드 간의 관계를 정의하여 복잡한 작업 흐름을 간결하게 관리할 수 있습니다. 이를 통해 복잡한 시스템을 효율적으로 설계하고 관리할 수 있는 장점이 있습니다.

#### 이번 프로젝트에서 LangGraph를 사용한 이유는 다음과 같은 장점들 때문입니다.

1. 직관적인 워크플로우 설계:
   LangGraph는 각 단계를 독립적인 노드로 설계하고, 이 노드들을 연결하여 복잡한 워크플로우를 직관적으로 구성할 수 있게 해줍니다. 이로 인해 시스템의 유지보수와 확장이 용이해집니다.
2. 조건부 엣지를 통한 유연한 흐름 제어:
   LangGraph의 조건부 엣지를 통해 특정 조건에 따라 워크플로우의 흐름을 동적으로 제어할 수 있습니다. 예를 들어, 레시피가 특정 질병이나 라이프스타일에 ‘부적합’으로 평가될 경우 워크플로우를 종료하고, ‘적합’으로 평가될 경우에만 다음 단계로 진행할 수 있습니다.
3. 타입 안정성과 데이터 구조 관리:
   Python의 타입 지정 기능과 결합하여 각 노드에서 처리되는 데이터 구조를 명확하게 정의할 수 있습니다. 이는 코드의 가독성을 높이고, 데이터 구조에 대한 명확한 이해를 제공하여 실수를 줄이고 시스템의 안정성을 높이는 데 기여합니다.
4. 시각화 기능:
   LangGraph의 시각화 기능을 통해 복잡한 워크플로우를 직관적으로 파악할 수 있으며, 팀 내에서 프로젝트 구조를 쉽게 공유할 수 있습니다.
5. 유연성과 확장성:
   복잡한 로직을 독립적인 노드로 분리하여 관리할 수 있기 때문에, 시스템의 유연성과 확장성을 크게 향상시킬 수 있습니다. 새로운 기능을 추가하거나 기존 기능을 수정하는 작업이 수월해집니다.

---

> 이번 프로젝트에서는 RAG 기반의 레시피 키워드 도출 시스템을 구현했습니다. 이 시스템은 특정 레시피가 당뇨와 같은 질병 또는 저탄수화물 다이어트와 같은 라이프스타일에 적합한지를 평가하고, 적합한 키워드를 도출합니다. LangGraph를 사용하여 이 시스템을 설계하고 구현했으며, 이를 통해 복잡한 로직을 효율적으로 관리할 수 있었습니다.

### Step1. 레시피 정보 가져오기

\`\`\`python
from step1 import fetch_recipe_with_nutrients

# step1. 레시피 정보를 데이터베이스에서 가져오는 노드
def fetch_recipe(state: GraphState) -> GraphState:
    recipe_id = state.get('recipe_id')
    # 데이터베이스에서 레시피 정보를 가져오는 로직
    recipe_info = fetch_recipe_with_nutrients(recipe_id)
    return GraphState(
        recipe_info=recipe_info,
    )
\`\`\`

첫 번째 단계는 데이터베이스에서 특정 레시피의 영양 정보를 가져오는 것입니다. 이 단계는 fetch_recipe 노드를 통해 구현되었습니다. SQLAlchemy를 사용하여 레시피와 관련된 데이터를 데이터베이스에서 가져온 후, GraphState 객체에 저장합니다.

### Step2. 질병 및 라이프스타일 정보 검색

\`\`\`py
from step2 import search_disease_info, search_lifestyle_info

# step2. 질병 관련 정보를 검색하는 노드
def disease_doc_info_search(state: GraphState) -> GraphState:
    keywords = state.get('keywords')
    # 질병 관련 식이요법 정보를 검색하는 로직
    disease_info = search_disease_info(keywords['disease'])
    return GraphState(
        recipe_info=state.get('recipe_info'),
        disease_info=disease_info,
        keywords=keywords
    )

# step2. 라이프스타일 관련 정보를 검색하는 노드
def lifestyle_info_search(state: GraphState) -> GraphState:
    keywords = state.get('keywords')
    # 라이프스타일 관련 식이요법 정보를 검색
    lifestyle_info = search_lifestyle_info(keywords['lifestyle'])
    return GraphState(
        recipe_info=state.get('recipe_info'),
        lifestyle_info=lifestyle_info,
        keywords=keywords
    )
\`\`\`

두 번째 단계는 Pinecone을 사용하여 특정 질병과 라이프스타일에 대한 정보를 검색하는 것입니다. 검색 결과는 LLM이 레시피의 적합성을 평가하는 데 사용됩니다. search_disease_info와 search_lifestyle_info 함수를 통해 각 질병 및 라이프스타일에 대한 정보를 검색합니다.
질병과 라이프스타일은 검색어나 프롬프트 스타일이 달라지기때문에 한번에 처리하는것보다는 병렬처리가 더 결과의 정확성을 올릴수 있을거라 판단하였습니다.

### step3. 질병에 대한 레시피의 적합성을 평가하는 노드

\`\`\`py
def assess_disease_suitability_node(state: GraphState) -> GraphState:
    recipe_info = state.get('recipe_info')
    disease_info = state.get('disease_info')

    # 질병에 대한 적합성 평가 로직
    disease_assessment = assess_disease_suitability(recipe_info, disease_info)

    # '부적합'인 경우 END로 종료
    suitability_status = check_suitability(disease_assessment)
    if suitability_status == '부적합':
        return END

    return GraphState(
        recipe_info=recipe_info,
        disease_info=disease_info,
        disease_assessment=disease_assessment,
        lifestyle_info=state.get('lifestyle_info'),
        lifestyle_assessment=state.get('lifestyle_assessment')
    )

\`\`\`

### step4.적합성 평가

\`\`\`py
from step3 import assess_disease_suitability, assess_lifestyle_suitability

# step3. 질병에 대한 레시피의 적합성을 평가하는 노드
def assess_disease_suitability_node(state: GraphState) -> GraphState:
    recipe_info = state.get('recipe_info')
    disease_info = state.get('disease_info')

    # 질병에 대한 적합성 평가 로직
    disease_assessment = assess_disease_suitability(recipe_info, disease_info)

    # '부적합'인 경우 END로 종료
    suitability_status = check_suitability(disease_assessment)
    if suitability_status == '부적합':
        return END

    return GraphState(
        recipe_info=recipe_info,
        disease_info=disease_info,
        disease_assessment=disease_assessment,
        lifestyle_info=state.get('lifestyle_info'),
        lifestyle_assessment=state.get('lifestyle_assessment')
    )

# step3. 라이프스타일에 대한 레시피의 적합성을 평가하는 노드
def assess_lifestyle_suitability_node(state: GraphState) -> GraphState:
    recipe_info = state.get('recipe_info')
    lifestyle_info = state.get('lifestyle_info')

    # 라이프스타일에 대한 적합성 평가 로직
    lifestyle_assessment = assess_lifestyle_suitability(recipe_info, lifestyle_info)

    # '부적합'인 경우 END로 종료
    suitability_status = check_suitability(lifestyle_assessment)
    if suitability_status == '부적합':
        return END

    return GraphState(
        recipe_info=recipe_info,
        lifestyle_info=lifestyle_info,
        lifestyle_assessment=lifestyle_assessment,
        disease_info=state.get('disease_info'),
        disease_assessment=state.get('disease_assessment')
    )
    )
\`\`\`

### Step5. 키워드 추출 및 저장

\`\`\`py
from step4 import create_guide, extract_keywords

# step4. 키워드를 추출하고, 평가 결과를 데이터베이스에 저장하는 노드
def extract_keyword_and_save(state: GraphState) -> GraphState:
    recipe_info = state.get('recipe_info')
    disease_assessment = state.get('disease_assessment')
    lifestyle_assessment = state.get('lifestyle_assessment')
    guide = create_guide(disease_assessment, lifestyle_assessment)
    keywords = extract_keywords(recipe_info)
    save_assessment_keywords_and_guide_to_db(state.get('recipe_id'), keywords, guide)
    return GraphState(
        recipe_info=recipe_info,
        keywords=keywords,
        guide=guide
    )
\`\`\`

마지막 단계에서는 적합성 평가 결과를 바탕으로 키워드를 추출하고, 이를 데이터베이스에 저장합니다. 이 단계는 extract_keyword_and_save 노드를 통해 구현되었습니다. 평가 결과에서 ‘적합’으로 평가된 항목만 키워드로 추출됩니다.

---

### 시스템 설계

#### 2-1. 타입지정

프로젝트의 타입 안성을 위해 Python의 TypedDict를 사용하여 각 노드에서 처리되는 데이터의 타입을 명확히 정의했습니다. 이렇게 함으로써, 각 노드가 입력과 출력으로 사용하는 데이터의 구조를 명확하게 유지할 수 있었습니다.

\`\`\`py
class GraphState(TypedDict):
    recipe_info: Optional[Dict]
    keywords: Optional[List[str]]
    disease_info: Optional[Dict]
    lifestyle_info: Optional[Dict]
    disease_assessment: Optional[str]
    lifestyle_assessment: Optional[str]
    guide: Optional[str]
    lifestyle_reason: Optional[str]
    disease_reason: Optional[str]
\`\`\`

GraphState는 각 노드에서 주고받는 데이터의 구조를 정의합니다. 이 구조는 시스템의 유지보수성을 높이고, 코드의 가독성을 향상시키는 역할을 합니다.

#### 2-2. 노드 설정 및 연결

LangGraph를 사용하여 시스템을 노드 기반으로 설계했습니다. 각 노드는 독립적으로 동작하며, 특정 작업을 수행하는 데 필요한 로직을 포함하고 있습니다. 노드 간의 연결은 그래프 구조로 이루어져 있어, 각 노드의 결과에 따라 다음 노드가 실행되도록 설계되었습니다.

\`\`\`py
# 노드 등록 및 연결 설정 (병렬화)
graph.add_node('fetch_recipe', fetch_recipe)
graph.add_node('disease_doc_info_search', disease_doc_info_search)
graph.add_node('lifestyle_info_search', lifestyle_info_search)
graph.add_node('extract_keyword_and_save', extract_keyword_and_save)
graph.add_node('assess_disease_suitability', assess_disease_suitability_node)
graph.add_node('assess_lifestyle_suitability', assess_lifestyle_suitability_node)

# 노드간의 연결 설정
graph.add_edge('fetch_recipe', 'disease_doc_info_search')
graph.add_edge('fetch_recipe', 'lifestyle_info_search')
\`\`\`

위 코드에서 볼 수 있듯이, 레시피 정보는 fetch_recipe 노드를 통해 가져온 후, 병렬로 질병 정보(disease_doc_info_search)와 라이프스타일 정보(lifestyle_info_search)를 검색하도록 설정했습니다. 그 후, 두 정보의 적합성을 평가하고(assess_disease_suitability_node, assess_lifestyle_suitability_node), 최종적으로 키워드 도출 및 저장 작업을 수행합니다(extract_keyword_and_save).

> LangGraph를 사용한 이번 워크플로우 구현은 복잡한 RAG 기반 시스템을 효율적으로 관리할 수 있게 해주는 좋은 예시입니다. 각 단계가 독립적인 노드로 관리되기 때문에 로직의 변경이나 확장이 용이하며, 조건부 엣지 설정을 통해 시스템의 유연성과 효율성을 극대화할 수 있었습니다. 특히, 워크플로우 시각화 기능은 개발자와 비개발자 모두가 시스템의 흐름을 쉽게 이해할 수 있게 해주며, 디버깅과 최적화 작업을 더 원활하게 할 수 있습니다.

#### 2-3. 조건부엣지 설정

\`\`\`python
# 조건부 엣지 설정
graph.add_conditional_edges(
    'assess_disease_suitability',
    check_suitability, # 적합,부적합 확인함수
    {
        '적합': 'extract_keyword_and_save',
        '부적합': END
    }
)
\`\`\`

조건부 엣지를 사용하여 노드 간의 흐름을 동적으로 제어했습니다. 예를 들어, 레시피가 특정 질병이나 라이프스타일에 부적합한 것으로 평가되면, 해당 워크플로우를 종료하고, 적합한 경우에만 다음 단계로 진행하도록 설정했습니다.

### workflow 시각화

LangGraph를 통해 구현한 워크플로우를 시각화하여 전체적인 흐름을 직관적으로 파악할 수 있도록 했습니다. 이는 디버깅 및 최적화 작업을 용이하게 하며, 프로젝트의 구조를 이해하는 데 큰 도움이 됩니다.

\`\`\`py
# 그래프 시각화
try:
    img = Image(app.get_graph(xray=True).draw_mermaid_png())

    with open("graph.png", "wb") as f:
        f.write(img.data)

    display(Image("graph.png"))
except:
    pass
\`\`\`

위 코드에서는 워크플로우를 Mermaid 형식으로 시각화하여 저장 및 출력할 수 있도록 구현했습니다. 이를 통해 프로젝트의 실행 흐름을 시각적으로 표현할 수 있어, 개발 및 유지보수 작업을 한층 수월하게 합니다.
아래는 시각화저장된 표입니다.

![](https://velog.velcdn.com/images/looa0807/post/9818445e-6ffb-4433-87d0-d064f3b66e88/image.png)

이 프로젝트는 복잡한 AI 기반 시스템을 설계하고 관리하는 데 있어 LangGraph의 강력한 기능을 실질적으로 보여주는 사례입니다. 추후 LangSmith나 RAGAS와 같은 라이브러리를 이용하여 RAG의 정확성을 테스트 해볼수 있게끔 하는 노드를 추가할 예정에 있습니다.
`;export{e as default};
