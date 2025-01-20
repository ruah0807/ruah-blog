# 롸그 프로젝트

## Begin

```bash
# 설치
corepack enable
pnpm create vite@latest

# 의존성 설치
pnpm i

# 실행
pnpm run dev
```

## Stack

이 프로젝트는 다음과 같은 기술 스택을 사용하여 개발되었습니다:

- **React**: 사용자 인터페이스를 구축하기 위한 JavaScript 라이브러리입니다.
- **TypeScript**: JavaScript의 슈퍼셋으로, 정적 타입을 지원하여 코드의 안정성과 가독성을 높입니다.
- **Vite**: 빠른 개발 환경을 제공하는 빌드 도구로, 모듈 번들링과 핫 모듈 교체(HMR)를 지원합니다.
- **pnpm**: 빠르고 효율적인 패키지 매니저로, 의존성 설치 및 관리에 사용됩니다.

### Environment

- **Node.js**: JavaScript 런타임 환경으로, 서버 측 개발 및 빌드 도구 실행에 사용됩니다.
- **Vite 개발 서버**: 빠른 개발 서버로, 실시간으로 변경 사항을 반영하여 개발 생산성을 높입니다.

### Build and Deploy

- **Vite 빌드 도구**: 최적화된 프로덕션 빌드를 생성하여 배포에 사용됩니다.
- **GitHub Actions**: 자동으로 빌드 및 배포를 수행하는 프로세스입니다.
