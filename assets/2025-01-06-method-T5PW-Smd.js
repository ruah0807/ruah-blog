const e=`---
title: Reshaping, stacking, squeezing, and unsqueezing, permute
---

> PyTorch에서 텐서를 조작하는 것은 딥 러닝 및 기계 학습에서 필수적인 작업. 텐서의 모양을 변경하거나 차원을 추가/제거하는 작업은 데이터를 원하는 형식으로 준비하거나 네트워크에 입력하기 위해 필요하다.
> 텐서 조작은 PyTorch의 기본 기능 중 하나이며, 데이터를 모델에 입력하거나 분석할 때 필수적.
> reshape, stack, squeeze, unsqueeze, view, permute 등의 함수는 텐서의 모양과 차원을 효율적으로 조정할 수 있게 도와주는 함수이다.

- **Reshaping** : 입력 텐서를 정의된 형태로 변경하는 것.(an input tensor to a defined shape
- **View** : 입력 텐서의 형태를 정의된 형태로 반환하면서 동일한 메모리를 유지하는 것(Return a view of an input tensor of certain shape but keep the same memory as the original tensor)
- **Stacking** : 여러 텐서를 위로 쌓거나 옆으로 쌓는 것.(combine multiple tensors on top of each other(vstack) or side by side (hstack))
- **Squeezing** : 텐서에서 모든 \`1\` 차원을 제거하는 것.(remove all \`1\` dimensions from a tensor)
- **Unsqueezing** : 텐서에 \`1\` 차원을 추가하는 것.(add a \`1\` dimension to a tensor)

\`\`\`py
# create a tensor
import torch
x = torch.arange(1., 10.)
x, x.shape
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/3f58c264-ef39-48d4-a9fc-4a07dc4cd10e/image.png)

### 1. Reshaping 텐서 재구성

- \`torch.reshape()\` 함수는 텐서의 크기(모양)를 변경합니다. 하지만 변경된 크기(모양)는 원래 텐서의 **원소 개수**와 호환되어야 한다.

#### 호환되는 모양

\`\`\`py
# Add an extra dimension
#  모양 변경 (1 x 9) : 기존의 원소 size가 9이기 때문에
x_reshaped = x.reshape(1, 9)
x_reshaped, x_reshaped.shape
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/f18686ab-c89c-4079-9dac-363b2d91312d/image.png)

⚠ 주의: 텐서를 재구성하려면 새 모양이 기존 원소 수와 호환되어야 함.

#### 호환되지 않는 모양

\`\`\`py
# 호환되지 않는 모양 (오류 발생)
torch.reshape(x, (1, 7))  # 원소 수 불일치
\`\`\`

### 2. View (Reshape와 유사하지만 메모리를 공유)

- \`torch.view()\` 함수는 \`reshape()\`와 비슷하지만, 원래 텐서와 메모리를 공유한다. 즉, \`view()\`를 사용해 생성된 텐서를 수정하면 원래 텐서도 변경.

\`\`\`py
# change the view
z = x.view(1,9)
z, z.shape
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/b7147ecd-2a5d-4c29-a595-fa582289b46a/image.png)

\`\`\`py
# Changing 'z' changes 'x'  = because a view of a tensor shares the same memory as the original input
# : z와 x는 동일한 메모리를 공유하기 때문에 z를 변경하면 x도 반영됨.
z[:,0] = 5 # z의 0번째 열을 5로 변경
z, x # z를 변경하면 x도 반영됨.
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/47c759fa-07b2-41ff-b188-5abe6b574fb0/image.png)

### 3. Stacking 텐서 결합

\`torch.stack()\` 함수는 여러 텐서를 새로운 차원에 따라 결합한다. 이를 통해 텐서를 수직, 수평 등 다양한 방식으로 쌓을 수있다.

\`\`\`py
# Stack tensors on top of each other
# Stack : 입력 텐서의 형태를 정의된 형태로 반환하면서 동일한 메모리를 유지하는 것.
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked, x_stacked.shape
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/8d1e50f9-ba65-4a66-ae35-5a2837aa066e/image.png)

### 4. Squeezing and unsqueezing 텐서 차원 제거 및 추가

- \`torch.squeeze()\` 함수는 텐서에서 크기가 1인 차원을 제거
- \`torch.unsqueeze()\` 함수는 텐서에 1 차원을 추가

#### Squeeze 텐서 차원 제거

\`\`\`py
# torch.squeeze() : 텐서에서 모든 1 차원을 제거하는 것.
print(f"Previeus tensor: {x_reshaped}")
print(f"Previeus tensor shape: {x_reshaped.shape}")

# Remove extra dimension
x_squeezed = x_reshaped.squeeze()
print(f"\\nNew tensor: {x_squeezed}")
print(f"New tensor shape: {x_squeezed.shape}")
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/5d48d613-405c-4481-ba07-db40da47652c/image.png)

#### unsqueeze 텐서 차원 추가

\`\`\`py
# torch.unsqueeze() : 텐서에 1 차원을 추가하는 것.
print(f"Previeus tensor: {x_squeezed}")
print(f"Previeus tensor shape: {x_squeezed.shape}")

x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\\nNew tensor: {x_unsqueezed}")
print(f"New tensor shape: {x_unsqueezed.shape}")
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/97398915-919f-4891-8803-3a51ab348a7a/image.png)

### 5. permute 텐서 차원 재배열

- \`torch.permute()\` 함수는 텐서의 차원을 재배열합니다. 이를 통해 텐서의 차원을 원하는 순서로 변경할 수 있습니다.

\`\`\`py
# torch.permute() : 텐서의 차원을 재배열하는 것.
x_original = torch.rand(size=(224, 224, 3)) # 높이, 너비, 색상 채널

# Permute the original tensor to rearrange the dimensions
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0 -> 1, 1 -> 2,  2 -> 0

print(f"Original shape: {x_original.shape}")
print(f"Permuted shape: {x_permuted.shape}")
\`\`\`

![](https://velog.velcdn.com/images/looa0807/post/daf9e4b0-1980-46af-991d-fc097308d319/image.png)
`;export{e as default};
