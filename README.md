# LiRPA XOR Demo (Python/CPP/MATLAB)

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

## Iterative Backward-only 실행 가이드

반복(iteration) 방식 backward-only 구현은 아래 파일에 있습니다.

- `lirpa_backward_only_iteration.py`

### 1) 기본 실행

```bash
python lirpa_backward_only_iteration.py
```

기본 실행 시 다음을 수행합니다.

- recursive backward-only와 iterative backward-only 결과 자동 비교
- iterative backward-only XOR 데모 실행

### 2) 비교 테스트만 실행

```bash
python -c "import lirpa_backward_only_iteration as it; it.compare_recursive_and_iterative(eps=0.02)"
```

### 3) 계층별 디버그 출력 모드

아래처럼 `debug=True`로 실행하면 layer별 pre-activation bound와
relaxation 파라미터(`alpha/beta`)가 출력됩니다.

```bash
python -c "import lirpa_backward_only_iteration as it; it.run_xor_demo_iterative(eps=0.02, debug=True)"
```

## C++ 실행 가이드

C++ 버전은 아래 파일에 있습니다.

- `lirpa_forward_backward_fc.cpp` (`std::vector` 기반 구현)
- `lirpa_forward_backward_fc_array.cpp` (배열 기반 구현)
- `lirpa_backward_only_iteration.cpp` (iterative backward-only, `std::vector` 기반 구현)
- `lirpa_backward_only_iteration_array.cpp` (iterative backward-only, 배열 기반 구현)

### 1) 빌드

Windows PowerShell + g++(MinGW 등) 기준:

```powershell
g++ -std=c++17 -O2 lirpa_forward_backward_fc.cpp -o lirpa_forward_backward_fc.exe
g++ -std=c++17 -O2 lirpa_forward_backward_fc_array.cpp -o lirpa_forward_backward_fc_array.exe
g++ -std=c++17 -O2 lirpa_backward_only_iteration.cpp -o lirpa_backward_only_iteration.exe
g++ -std=c++17 -O2 lirpa_backward_only_iteration_array.cpp -o lirpa_backward_only_iteration_array.exe
```

WSL Ubuntu + clang++ 기준:

```powershell
wsl clang++ -std=c++17 -O2 /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc.cpp -o /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_wsl
wsl clang++ -std=c++17 -O2 /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_array.cpp -o /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_array_wsl
wsl clang++ -std=c++17 -O2 /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration.cpp -o /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_wsl
wsl clang++ -std=c++17 -O2 /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_array.cpp -o /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_array_wsl
```

### 2) 기본 실행 (eps=0.02)

```powershell
./lirpa_forward_backward_fc.exe
./lirpa_forward_backward_fc_array.exe
./lirpa_backward_only_iteration.exe
./lirpa_backward_only_iteration_array.exe
```

WSL Ubuntu 바이너리 실행:

```powershell
wsl /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_wsl
wsl /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_array_wsl
wsl /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_wsl
wsl /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_array_wsl
```

### 3) epsilon을 바꿔 실행하기

실행 인자로 epsilon 값을 넣으면 됩니다.

```powershell
./lirpa_forward_backward_fc.exe 0.22
./lirpa_forward_backward_fc.exe 0.23
./lirpa_forward_backward_fc_array.exe 0.22
./lirpa_forward_backward_fc_array.exe 0.23
./lirpa_backward_only_iteration.exe 0.02 --debug
./lirpa_backward_only_iteration_array.exe 0.02 --debug
```

WSL Ubuntu 바이너리에서 epsilon/debug 옵션 사용 예시:

```powershell
wsl /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_wsl 0.22
wsl /mnt/d/Work/lang/python/crown/lirpa_forward_backward_fc_array_wsl 0.23
wsl /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_wsl 0.02 --debug
wsl /mnt/d/Work/lang/python/crown/lirpa_backward_only_iteration_array_wsl 0.02 --debug
```

권장 실험:

- `0.00`, `0.10`, `0.22`, `0.23` 순서로 실행
- 각 입력점별 `certified=True/False`를 비교
- forward/backward 결과가 어떻게 달라지는지 함께 확인

## 출력 해석 팁

각 입력점마다 다음 정보가 출력됩니다.

- 네트워크 점 예측값 (`network_output`)
- forward bound와 backward bound의 하한/상한
- XOR 정답 클래스 기준으로 인증 여부 (`certified=True/False`)


## MATLAB 실행 가이드

MATLAB 버전은 아래 파일에 있습니다.

- `lirpa_forward_backward_fc.m`

MATLAB 콘솔에서 실행:

```matlab
lirpa_forward_backward_fc_matlab
```

MATLAB 코드 내부의 `run_xor_demo(...)` 인자를 바꿔 epsilon 실험을 반복하면 동일한 경향을 확인할 수 있습니다.

XOR 정답이 1인 점은 보통 `lower bound > 0.5`, 정답이 0인 점은 `upper bound < 0.5` 조건으로 인증 여부를 판단합니다.
