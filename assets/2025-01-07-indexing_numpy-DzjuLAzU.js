const n=`---
title: Indexing & Numpy in PyTorch
---

## Select Data from tensor(Indexing)

> 텐서 인덱싱은 텐서에서 특정 요소나 데이터 범위를 선택하고 조작하는 방법. 다차원 데이터를 처리할 때 필수적인 기술이며, 데이터를 효과적으로 선택하고 활용할 수 있도록 돕는다.
> PyTorch의 텐서 인덱싱은 **NumPy 인덱싱**과 매우 유사하다.

\`\`\`py
# Create a tensor
import torch

x= torch.arange(1, 10).reshape(1,3,3)
x, x.shape
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/47137844-264e-4613-a632-ce760293ed2f/image.png)

크기 (1, 3, 3)는 다음을 의미한다:

- 텐서 1개
- 각 텐서에 3개의 행(row)
- 각 행에 3개의 요소

### Basic Indexing

요소 접근하기

인덱싱의 기본 형식: tensor [ 차원_1, 차원_2, ... ]

#### 첫 번째 텐서 (0번째 인덱스) 접근

\`\`\`py
print(x[0])
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/46050a7b-1047-4f11-bf50-c3a1a5a271c8/image.png)

#### 첫 번째 텐서의 첫 번째 행 접근

\`\`\`py
print(x[0][0])
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/e153ed90-3788-4ac8-8590-b87d0b1db619/image.png)

#### 첫 번째 텐서의 3 번째 행의 3 번째 요소 접근

\`\`\`py
print(x[0][2][2])
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/9935d01c-f302-413b-892c-8b887d9294b2/image.png)

### Advanced Indexing : Slicing

- 슬라이싱을 사용하면 특정 차원에서 여러 요소를 선택할 수 있다. 슬라이싱 구문은 Python의 표준 형식인 **start:end:step**을 따름.

\`\`\`py
# 모든 행을 선택하고 첫 번째 열만 가져오기
print(x[:, :, 0])  # 결과: [[1, 4, 7]]

# 첫 번째 텐서의 두 번째 행의 모든 요소 가져오기
print(x[0, 1, :])  # 결과: [4, 5, 6]

# 모든 행에서 마지막 열 가져오기
print(x[:, :, -1])  # 결과: [[3, 6, 9]]
\`\`\`

#### 인덱싱과 슬라이싱 결합

- 인덱싱과 슬라이싱을 결합하여 텐서의 특정 부분 추출이 가능하다.

\`\`\`py
# 첫 번째 텐서의 모든 행과 열 접근
print(x[0, :, :])
\`\`\`

\`\`\`py
# 첫 번째 텐서의 첫 두 행 접근
print(x[0, :2, :])
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/a5422ee2-8944-4c5a-bf85-7704ba84d2a6/image.png)

\`\`\`py
# 모든 행을 선택하고 두 번째 열만 가져오기
print(x[:, :, 1])  # 결과: [[2, 5, 8]]
\`\`\`

---

## PyTorch Tensor & NumPy

> PyTorch는 NumPy와 긴밀하게 연동된다. NumPy는 수치 계산을 위한 매우 인기 있는 라이브러리다. PyTorch는 NumPy 데이터로 시작하거나, NumPy 배열을 PyTorch 텐서로 변환하는 기능을 제공한다. 반대로 PyTorch 텐서를 다시 NumPy 배열로 변환할 수도 있다.

- Numpy에서 원하는 데이터는 PyTorch 텐서 -> \`torch.from_numpy(ndarray)\`
- Pytorch tensor -> Numpy -> \`torch.Tensor.numpy()\`

### 1. NumPy 배열에서 PyTorch 텐서로 변환하기

- PyTorch에는 NumPy 배열을 텐서로 변환할 수 있는 \`torch.from_numpy()\`라는 메서드가 있다.

\`\`\`py
# Numpy array to tensor
import torch
import numpy as np

array = np.arange(1.0, 8.0)

# warning : when converting from numpy => pytorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise
tensor = torch.from_numpy(array).type(torch.float32)
tensor, array
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/4643dfa6-1af0-4833-8aaa-3cc9eba90347/image.png)

#### NumPy와 PyTorch의 기본 데이터 타입

- NumPy의 기본 데이터 타입은 float64다.
- PyTorch의 기본 데이터 타입은 float32다.
- NumPy 배열을 PyTorch 텐서로 변환하면 NumPy의 데이터 타입이 유지된다.

\`\`\`py
print("NumPy 배열 데이터 타입:", array.dtype)  # float64
print("PyTorch 텐서 데이터 타입:", tensor.dtype)  # float64
\`\`\`

### 2. 데이터 타입 변경하기

NumPy에서 PyTorch로 데이터를 변환할 때 데이터 타입을 변경할 수도 있다.

\`\`\`py
# 데이터 타입을 float32로 변환
tensor = torch.from_numpy(array).type(torch.float32)
\`\`\`

### 3. NumPy 배열 값 변경 시 PyTorch 텐서의 변화

\`\`\`py
# Change the value of array, what will this do to \`tensor\`?
array = array +1
array, tensor
\`\`\`

- 아래와 같이 array는 변하지만 tensor값은 변하지 않는다.

![](https://velog.velcdn.com/images/looa0807/post/52281e3f-7f02-42e2-a543-23f4fd1a5fb1/image.png)

### 4. PyTorch 텐서에서 NumPy 배열로 변환하기

- Tensor를 Numpy로 변환후 tensor 값을 변경시에도 Numpy에 영향을 주지 않는다.
- \`.numpy()\` 메서드를 사용하여 변환

\`\`\`py
# Create Tensor to Numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/481ed112-d3c0-45cf-a07c-cfeff84a7856/image.png)

\`\`\`py
# Change the tensor, what happens to \`numpy_tensor\`?
tensor = tensor + 1
tensor, numpy_tensor
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/36b5193b-d34a-4475-ba37-65aba547dcbc/image.png)

### 정리

1. 기본 데이터 타입 차이
   - NumPy: float64
   - PyTorch: float32
   - 이 차이로 인해 연산 시 데이터 타입 문제나 오류가 발생할 수 있다. 필요하다면 데이터 타입을 명시적으로 설정한다.
2. 메모리 공유
   - NumPy에서 PyTorch로 변환된 데이터는 메모리를 공유하지 않는다.
   - 데이터를 변경해도 서로 영향을 미치지 않는다.
`;export{n as default};
