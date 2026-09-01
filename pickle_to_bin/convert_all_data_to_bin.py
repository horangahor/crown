
"""
pickle 데이터셋 -> crown_test_all_data.cu 용 이진 파일 변환기
사용법: python convert_all_data_to_bin.py [data_path] [no_test_files] [output_path]
출력: (N, input_dim) float64 이진 파일 (C++ fread 호환)
"""

import sys
import numpy as np

# 경로를 맨 앞(0)에 추가 <== 찾아보는 순위에 영향
sys.path.insert(0, 'C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/crown')

from data_split import load_test_rows

# 시스템 변수
data_path   = sys.argv[1] if len(sys.argv) > 1 else (
    '../wireless\mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle'
)
no_test_files = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# 데이터 파일은 쉘 or cmd에서 스크립트를 실행한 경로를 기준으로 함
output_path   = sys.argv[3] if len(sys.argv) > 3 else 'test_data_all.bin'

print(f'Loading data from: {data_path}')
print(f'no_test_files = {no_test_files}  =>  {20000 * no_test_files} samples')

X = load_test_rows(data_path, no_dataInFile=20000, no_test_files=no_test_files)
print(f'Loaded shape: {X.shape}  dtype: {X.dtype}')

# C++ 에서 double(float64) 로 읽으므로 <f8 (little-endian float64) 로 저장
X.astype('<f8').tofile(output_path)
print(f'Saved {X.shape[0]} x {X.shape[1]} float64 -> {output_path}')
print(f'File size: {X.nbytes} bytes ({X.nbytes / 1024 / 1024:.1f} MB)')
