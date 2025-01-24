const n=`---
title: PyTorch에서 랜덤 시드 사용하기
---

## Random Seed : 재현성(reproducibility)보장

> 재현성은 실험을 반복하거나 결과를 검증할때, 일관된 결과를 얻어야하기때문에 필요하다.

**신경망이 학습하는 방법:**
\`랜덤숫자 부터 시작 -> tensor 연산 수행 -> 데이터의 더 나은 표현 시도를 위한 랜덤 숫자 업데이트 -> 더 나은 랜덤 숫자 업데이트 -> 더 나은 랜덤 숫자 업데이트 -> ...\`
신경망의 무작위성을 줄이기 위해 그리고 파이토치는 **random seed**를 사용한다.
본질적으로 random seed는 'flavour' 또는 '무작위성의 종류'를 선택하는 것이다.

### Random Seed를 이용하는 이유

#### 1. 재현성 보장

- 머신러닝 실험에서 랜덤 초기화 값이 다르면 결과가 달라질 수 있다.
- 예를 들어:
  - 가중치 초기화
  - 데이터셋의 샘플링
  - 데이터 분할(훈련/검증/테스트)
- 랜덤 시드를 고정하면 동일한 입력 데이터와 설정으로 항상 같은 결과를 얻는다.

#### 2. 디버깅과 테스트 용이

- 코드에서 오류를 찾거나 성능 문제를 디버깅할 때, 동일한 랜덤 값으로 반복 실험을 할 수 있어서 편리하다.
- 테스트 중에 불확실성을 제거하고, 문제가 랜덤 값 때문인지 로직 때문인지 확실히 알 수 있다.

#### 3. 결과 비교 가능

- 여러 모델이나 알고리즘의 성능을 비교할 때, 랜덤 요소를 고정하면 공정한 비교가 가능하다.
- 예를 들어, 같은 데이터 분할을 사용하는지 보장.

### torch.manual_seed()

\`\`\`py
import torch
# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Random Tensor A: {random_tensor_A}")
print(f"\\nRandom Tensor B: {random_tensor_B}")
print(f"\\nRandom Tensor A == Random Tensor B: \\n{random_tensor_A == random_tensor_B}")
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/60d6c149-f696-4034-8b6f-199b306d2ae8/image.png)

위 결과와 같이 텐서는 랜덤으로 무작위 숫자가 선정이되기 때문에 재현할수가 없다.
해서, 아래와 같이 **Random Seed**를 활용하여 무작위 선정한 숫자들을 재현한다.

\`\`\`py
# Making some random but reproducible tensors
import torch
# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(f"Random Tensor C: {random_tensor_C}")
print(f"Random Tensor D: {random_tensor_D}")

print(f"\\nRandom Tensor C == Random Tensor D: \\n{random_tensor_C == random_tensor_D}")

\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/8a5587b9-c3e7-45d7-a30e-fae1e82edc92/image.png)

주의해야할 점은 각 텐서가 생성될때마다 RandomSeed를 다시 선언해주어야하는것이다. 만일 random_tensor_D위쪽에 RandomSeed를 선언하지 않았다면 또다른 랜덤 숫자가 생성되었을것이다.

---
`;export{n as default};
