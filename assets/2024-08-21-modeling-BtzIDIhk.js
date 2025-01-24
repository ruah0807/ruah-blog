const n=`---
title: 데이터 모델링(Data Modeling)
---

## 데이터 모델링이란?

데이터 모델링(Data Modeling)은 데이터 베이스의 설계의 중요한 단계로, 데이터를 구조화 하고 조직화하는 과정입니다. 이 과정에서는 현실 세계의 비즈니스 요구사항을 분석하고 이를 데이터베이스의 테이블, 열, 관계 등으로 표현하는 일을 합니다.

## 데이터 모델링의 주요 단계

### 1. 개념적 데이터 모델링 (Conceptual Data Modeling)

- 높은 수준에서 비즈니스 요구사항을 파악하고, 이를 엔티티(Entity), 속성(Attribute), 관계(Relationship) 등으로 표현
- 구체적인 데이터 베이스 시스템을 고려하지 않고, 비즈니스 자체에 초점

#### 개념적 데이터 모델링 예시 :

\`\`\`javascript
엔티티(Entity):
	고객(Customer): 쇼핑몰을 이용하는 사람들.
	제품(Product): 쇼핑몰에서 판매되는 물품.
	주문(Order): 고객이 쇼핑몰에서 구매하는 행위.
	장바구니(Cart): 고객이 구매를 위해 담아둔 제품 목록.
속성(Attribute):
	고객(Customer): 고객 ID, 이름, 이메일, 전화번호.
	제품(Product): 제품 ID, 이름, 가격, 재고 수량.
	주문(Order): 주문 ID, 주문 날짜, 배송 상태.
	장바구니(Cart): 장바구니 ID, 생성 날짜.
관계(Relationship):
	고객은 여러 주문을 할 수 있다. (1:N 관계)
	주문은 여러 제품을 포함할 수 있다. (N:M 관계)
	고객은 장바구니를 가질 수 있다. (1:1 관계)
	장바구니는 여러 제품을 담을 수 있다. (N:M 관계)
\`\`\`

### 2. 논리적 데이터 모델링 (Logical Data Modeling)

- 개념적 모델링을 바탕으로, 보다 구체적인 구조로 변환하는 단계
- 테이블, 컬럼, 데이터 타입 등을 정의하고, 각 엔티티 간의 관계를 명확히하는 단계

#### 논리적 데이터 모델링 예시 : 테이블(Table)

\`\`\`javascript
Customer (고객 테이블)
	customer_id (Primary Key)
    name
    email
    phone_number
Product (제품 테이블)
	product_id (Primary Key)
    name
    price
    stock_quantity
Order (주문 테이블)
	order_id (Primary Key)
    order_date
    shipping_status
    customer_id (Foreign Key, Customer와 관계)
Cart (장바구니 테이블)
	cart_id (Primary Key)
    created_date
    customer_id (Foreign Key, Customer와 관계)
    Order_Product (주문-제품 관계 테이블)
    order_id (Foreign Key, Order와 관계)
    product_id (Foreign Key, Product와 관계)
    quantity
\`\`\`

### 3. 물리적 데이터 모델링 (Physical Data Modeling)

- 논리적 모델을 실제 데이터베이스에 구현하기 위해 설계하는 단계
- 데이터 베이스 시스템의 특정기능(예 : 인덱스, 파티셔닝 등) 을 고려하여 최적화된 데이터 베이스 구조를 만든다.

#### 물리적 데이터 모델링 예시

\`\`\`javascript
인덱스(Index):
	Customer 테이블에서 email에 인덱스를 추가하여 검색 속도를 높임.
	Order 테이블에서 order_date에 인덱스를 추가하여 주문 기록을 시간순으로 빠르게 조회할 수 있게 함.
파티셔닝(Partitioning):
	Order 테이블을 주문 날짜를 기준으로 파티셔닝하여, 대규모 데이터를 관리하기 쉽게 함.
데이터 타입 최적화:
	phone_number를 VARCHAR(15)로 설정하여 전화번호의 길이를 고려함.
	price를 DECIMAL(10,2)로 설정하여 소수점을 포함한 가격을 정확하게 표현함.

\`\`\`

## 데이터 모델링의 목적

- 데이터 무결성(Integrity)와 일관성(Consistency)을 보장
- 비즈니스 요구사항을 데이터베이스에 잘 반영하여, 효울적으로 데이터를 저장하고 조회할 수 있게함.
- 향후 시스템 확장 및 유지보수가 용이하게 만듬.

쉽게말해, 데이터 모델링은 우리가 데이터를 어떻게 구조화할지 계획하고, 이를 구현하는 과정입니다. 이 과정을 통해 데이터를 보다 효율적으로 관리하고 활용할 수 있습니다.
`;export{n as default};
