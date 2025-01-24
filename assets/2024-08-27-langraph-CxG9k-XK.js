const n=`---
title: LangGraph 사용하기
---

## State 정의

![](https://velog.velcdn.com/images/looa0807/post/875a8ea0-a02c-4914-a156-d0a4ff7231ee/image.png)

- Annotated: 타입 힌트에 추가적인 메타데이터(예: 주석)를 부여하기 위한 기능
- Sequence: 순서가 있는 컬렉션(리스트, 튜플 등)을 일반화한 타입
- operator: 다양한 연산자를 함수로 제공하는 모듈.
  - operator.add: 두 값을 더하는 함수입

## 노드 정의

![](https://velog.velcdn.com/images/looa0807/post/5141e111-eda8-4a1f-a28a-557c20d943ab/image.png)![](https://velog.velcdn.com/images/looa0807/post/613fe013-77bd-497f-9c50-5d2fed83dfbe/image.png)

- 각 sequence마다 갱신되는 위치가 다름

## 그래프 정의

![](https://velog.velcdn.com/images/looa0807/post/a1156b21-b744-4687-8fab-e315cf1f1549/image.png)

- StateGraph 선언 후

#### 노드 추가

- 왼쪽 : node 이름
- 오른쪽 : 위에서 정의한 함수 이름

![](https://velog.velcdn.com/images/looa0807/post/6ea24492-dd7a-4256-9c1c-f6699c41416b/image.png)

#### 노드 연결

- 왼쪽 : 보내는 node
- 오른쪽 : 받는 node

![](https://velog.velcdn.com/images/looa0807/post/08f0f582-bfa6-49ff-bdd9-261783dbad37/image.png)

#### 조건부 엣지 추가

![](https://velog.velcdn.com/images/looa0807/post/8a89c318-c482-459b-bb6a-74b12f196733/image.png)

- 시작지점 설정
- 메모리 저장소 설정
- 컴파일하는 단계에 메모리 저장

![](https://velog.velcdn.com/images/looa0807/post/083f9061-80ba-4b46-89c1-62eae9b3a1af/image.png)

#### 그래프 시각화

![](https://velog.velcdn.com/images/looa0807/post/067d3a91-4b5a-4dfe-81c3-e7d1e7d6818f/image.png)

- LangGraph를 이용한 RAG pipeline
`;export{n as default};
