# LiRPA XOR Demo (Python/MATLAB)

이 저장소는 완전연결 신경망(은닉층 ReLU, 출력층 Sigmoid)에 대해 LiRPA(CROWN 계열) forward/backward bound를 계산하는 교육용 예제를 제공합니다.

## 핵심 관찰 (XOR 게이트)

XOR 네 코너 입력 `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`에 대해 다음이 관찰됩니다.

- 교란 반경 `epsilon`이 `0 ~ 0.22`이면 4가지 입력 모두 certified 됩니다.
- `epsilon = 0.23`이면 4가지 중 2가지 입력은 certified되지 않습니다.

즉, epsilon을 조금만 키워도 인증 가능 여부가 바뀌는 경계가 존재하므로, 값을 바꿔가며 결과를 확인해보는 것이 중요합니다.

## Python 실행 가이드

### 1) 기본 실행

아래 파일이 데모 진입점입니다.

- `lirpa_forward_backward_fc.py`

기본값(`eps=0.02`)으로 바로 실행:

```bash
python lirpa_forward_backward_fc.py
```

### 2) epsilon을 바꿔 실행하기

`run_xor_demo(eps=...)`를 직접 호출하면 원하는 epsilon으로 실험할 수 있습니다.

```bash
python -c "import lirpa_forward_backward_fc as d; d._self_test_relaxations(); d.run_xor_demo(eps=0.22)"
python -c "import lirpa_forward_backward_fc as d; d._self_test_relaxations(); d.run_xor_demo(eps=0.23)"
```

권장 실험:

- `eps=0.00`, `0.10`, `0.22`, `0.23` 순서로 실행
- 각 입력점별 `certified=True/False`를 비교
- forward/backward 결과가 어떻게 달라지는지 함께 확인

## MATLAB 실행 가이드

MATLAB 버전은 아래 파일에 있습니다.

- `lirpa_forward_backward_fc.m`

MATLAB 콘솔에서 실행:

```matlab
lirpa_forward_backward_fc_matlab
```

MATLAB 코드 내부의 `run_xor_demo(...)` 인자를 바꿔 epsilon 실험을 반복하면 동일한 경향을 확인할 수 있습니다.

## 출력 해석 팁

각 입력점마다 다음 정보가 출력됩니다.

- 네트워크 점 예측값 (`network_output`)
- forward bound와 backward bound의 하한/상한
- XOR 정답 클래스 기준으로 인증 여부 (`certified=True/False`)

XOR 정답이 1인 점은 보통 `lower bound > 0.5`, 정답이 0인 점은 `upper bound < 0.5` 조건으로 인증 여부를 판단합니다.
