import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np

# 모듈을 찾을 수 있도록 jnunnv/crown 경로를 시스템 경로에 추가 <== 모듈을 찾을 수 있게 함 (시스템 환경 변수같은 역할)
sys.path.append("C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/crown")
try:
    # pyrefly: ignore [missing-import]
    from data_split import load_test_rows
except ImportError:
    print("경고: data_split 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")

def main():
    # 데이터 경로 (기본값 설정)
    default_data_path = "C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/Pickle/mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D.pickle"
    
    # 인자로 받은 경로가 있으면 사용, 없으면 기본값 사용
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path

    print(f"데이터 로딩 중: {data_path}")
    
    # Pickle 파일에서 데이터 불러오기 (해당 데이터셋은 샘플이 들어있는 파일 여러개가 붙어있는 구조)
    # no_dataInFile : 원본파일 1개당 들어있는 샘플 수
    # no_test_files : test로 사용할 원본파일의 수 (1이면 20000개, 2이면 40000개)
    try:
        X = load_test_rows(data_path, no_dataInFile=20000, no_test_files=1)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return
    
    # X는 20000 , 256
    # 테스트할 첫 번째 데이터 1개 (x0) 추출
    x0 = X[0]
    print(f"테스트 데이터 1개 추출 완료. 크기: {x0.shape}") # 보통 (256,) 1차원 배열
    
    # 바이너리(.bin)로 저장
    # C++에서 double 배열로 읽을 수 있도록 astype("<f8") 사용
    out_path = "test_data_1.bin"
    x0.astype("<f8").tofile(out_path)
    
    print(f"성공적으로 변환 완료! C++ 용 파일: {out_path}")

if __name__ == "__main__":
    main()
