# CROWN GPU Multi-Stream 비동기 배칭 구현 및 벤치마크 계획

이 문서는 `crown_next_research_roadmap.md`의 **아이디어 1: 배치 처리(Batching) 및 동시성(Concurrency) - [방법 A] CUDA Multi-Stream 비동기 배칭**을 CROWN 검증 파이프라인에 적용하고, `crown_gemini_test.exe`로 빌드하여 성능 및 정확도를 측정·기록하는 작업 계획입니다.

---

## 1. 배경 및 문제 정의

- **현재 상태**: 단일 샘플($B=1$) 순차 처리 방식. RTX 4070 SUPER(56개 SM, 7,168 코어) 환경에서 단일 신경망 노드 연산(입력 256, 은닉 491, 출력 16)은 수백 개 스레드만 사용하여 대부분의 SM이 유휴(Idle) 상태에 머묾 (**GPU Occupancy < 15%**).
- **로드맵 1번 방법 ([방법 A] CUDA Multi-Stream 비동기 배칭)**:
  - 기존의 단일 샘플 CROWN 커널 로직은 그대로 보존.
  - $B$개의 독립적인 non-blocking `cudaStream_t`와 스트림별 작업 메모리 풀(`StreamContext` / `StreamPool`)을 구성.
  - 샘플 $B$개를 각 스트림에 비동기로 동시 큐잉(`network_forward_async`, `lirpa_backward_bound_async`, `cudaMemcpyAsync`).
  - GPU 하드웨어 작업 분배기(CWD: Central Work Distributor)가 서로 다른 스트림의 커널들을 비어있는 SM에 동시 스케줄링하여 처리량(Throughput)을 극대화.

---

## 2. 사용자 검토 항목 (User Review Required)

> [!IMPORTANT]
> **실행 데이터셋 규모 및 벤치마크 시간 안내**
> - 전체 데이터셋은 $N = 40,000$ 샘플 $\times$ 23개 $\epsilon$ 값 = 총 920,000회 CROWN 연산입니다.
> - 기존 단일 스트림 기준 약 1,500초(25분)가 소요되었습니다.
> - Multi-Stream 배칭($B=16$) 적용 시 예상 소요 시간은 약 5~8분 내외입니다.
> - 검증 진행 시 먼저 소규모(예: 1,000 샘플 또는 1개 eps)로 1차 정합성(100% 동일한 T/F/ratio)을 확인한 뒤 전체 40,000 샘플 벤치마크를 수행하여 결과를 기록할 예정입니다.

---

## 3. 제안 아키텍처 및 구현 내용

### 3-1. Multi-Stream 메모리 및 스트림 아키텍처

```
+--------------------------------------------------------------------------------+
|                             GPU Device Memory                                  |
|                                                                                |
|  [공유 불변 가중치 메모리 (Shared Read-Only Memory)]                           |
|  d_weight[0..2], d_weight_pos[0..2], d_weight_neg[0..2], d_bias[0..2] (~48 MB) |
|                                                                                |
|  [Stream 0 Pool (~16MB)]       [Stream 1 Pool (~16MB)]  ...  [Stream B-1 Pool] |
|  - d_fwd_mat, d_fwd_vec        - d_fwd_mat, d_fwd_vec        ...               |
|  - d_bwd_mat, d_bwd_vec        - d_bwd_mat, d_bwd_vec                          |
|  - d_infer, d_final_bounds     - d_infer, d_final_bounds                       |
+--------------------------------------------------------------------------------+
       ^                                ^                             ^
       | stream 0                       | stream 1                    | stream B-1
   [Sample 0 Async Queue]          [Sample 1 Async Queue]        [Sample B-1 Async]
```

- **메모리 사용량**: 스트림당 약 16MB. $B=16$ 기준 약 256MB로 RTX 4070 SUPER의 12GB VRAM 중 약 2%만 사용하여 매우 안전.
- **스트림 생성**: `cudaStreamCreateWithFlags(&streams[b], cudaStreamNonBlocking)`로 레거시 기본 스트림과의 동기화 간섭 차단.
- **비동기 호스트 버퍼**: 스트림별 pinned host memory(`cudaHostAlloc`)를 활용하여 H2D 및 D2H 전송의 CPU 블로킹 오버헤드 제거.

### 3-2. 소스 코드 구성

#### [NEW] [`my_lirpa_multistream.cuh`](file:///c:/Users/user/Desktop/cuda/crown/testing(present)/my_lirpa_multistream.cuh)
- 기존 `my_lirpa_optimizing.cu`의 커널 로직을 100% 재사용하되, `cudaStream_t stream` 및 `StreamPool& pool`을 인자로 받는 비동기 함수 추가:
  - `network_forward_async(...)`
  - `lirpa_forward_bound_impl_async(...)`
  - `backward_bound_gpu_async(...)`
  - `lirpa_backward_bound_async(...)`
- 공유 불변 가중치(`prepare_network_on_gpu`)는 1회만 초기화하여 모든 스트림이 동시 참조.

#### [NEW] [`crown_gemini_test.cu`](file:///c:/Users/user/Desktop/cuda/crown/testing(present)/crown_gemini_test.cu)
- 배치 단위 비동기 실행 루프 구현:
  ```cpp
  for (int i = 0; i < N; i += BATCH_SIZE) {
      int cur_b = std::min(BATCH_SIZE, N - i);
      for (int b = 0; b < cur_b; ++b) {
          enqueue_sample_async(dataset[i + b], eps, streams[b], pools[b], host_buffers[b]);
      }
      cudaDeviceSynchronize();
      for (int b = 0; b < cur_b; ++b) {
          bool cert = certify_topk(host_buffers[b].y0, host_buffers[b].lb, host_buffers[b].ub, k);
          if (cert) ++t_count; else ++f_count;
      }
  }
  ```
- 명령행 인자 지원:
  - `method`: 1 (backward, 기본값) 또는 0 (forward)
  - `batch_size`: 기본 16 (1, 4, 8, 16, 32 등 설정 가능)
  - `max_samples`: 전체 N 또는 특정 개수
  - `network_path`, `data_path`, `output_csv`
- 단계별 시간(추론, Bound, 검증, 총 시간) 및 Throughput(samples/sec) 측정.
- 기존 `results_cuda_backward.csv` 및 `timing_log_crown.csv` 포맷과 호환되는 CSV 저장.

---

## 4. 빌드 및 실행 명세

### 4-1. 빌드 명령
```powershell
nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728" -arch=sm_89 -O2 crown_gemini_test.cu -o crown_gemini_test.exe
```

### 4-2. 검증 단계
1. **단위 검증 (1개 샘플 `test_data_1.bin`)**:
   - `crown_gemini_test.exe 1 MODEL_PATH test_data_1.bin`
   - 기존 `crown_test_all_data.exe` 결과와 23개 $\epsilon$별 T/F가 완벽히 일치하는지 확인.
2. **배치 크기 확장성 검증 ($B=1$ vs $B=4$ vs $B=16$)**:
   - 동일 샘플 수(예: 1,000개 샘플)에서 배칭에 따른 실행 시간 및 가속비(Speedup) 확인.
3. **전체 데이터셋($N=40,000$, 23 $\epsilon$) 최종 실행**:
   - `crown_gemini_test.exe 1 MODEL_PATH test_data_all.bin 8 results_gemini.csv`
   - 최종 실행 시간, 초당 처리량, 정합성(기존 baseline CSV 대비 일치율 100%) 확인.
4. **결과 기록**:
   - 실행 통계, 가속비, $\epsilon$별 T/F 결과를 `walkthrough.md` 및 timing CSV에 기록.
