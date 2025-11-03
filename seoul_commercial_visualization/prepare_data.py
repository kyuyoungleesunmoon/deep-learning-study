"""
서울시 상권분석 데이터 준비 스크립트
Seoul Commercial District Data Preparation Script

이 스크립트는 교육용 샘플 데이터를 생성합니다.
실제 서울시 상권분석서비스 데이터는 다음에서 다운로드할 수 있습니다:
https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_seoul_commercial_data():
    """
    서울시 상권분석 데이터와 유사한 샘플 데이터 생성
    
    실제 데이터셋의 주요 컬럼을 포함:
    - 기준년월코드, 상권구분코드명, 상권코드명
    - 서비스업종코드명, 월요일매출금액, ..., 일요일매출금액
    - 연령대별 매출 등
    """
    
    np.random.seed(42)
    
    # 기본 설정
    n_records = 1000
    
    # 시간 데이터
    start_date = datetime(2023, 1, 1)
    quarters = pd.date_range(start=start_date, periods=4, freq='Q')
    year_months = [q.strftime('%Y%m') for q in quarters]
    
    # 서울시 구 및 상권
    districts = ['강남구', '서초구', '송파구', '강동구', '종로구', '중구', 
                 '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구',
                 '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구',
                 '양천구', '강서구', '구로구', '금천구', '영등포구', '동작구', '관악구']
    
    commercial_zones = ['골목상권', '발달상권', '전통시장', '관광특구']
    
    # 업종
    industries = ['한식음식점', '커피/음료', '편의점', '의류', '화장품', 
                  '병원', '학원', '부동산', '헬스/피트니스', '미용실',
                  '제과점', '치킨전문점', '주점', '일반의류', '스포츠/아웃도어']
    
    # 연령대
    age_groups = ['10대', '20대', '30대', '40대', '50대', '60대이상']
    
    data = []
    
    for i in range(n_records):
        year_month = np.random.choice(year_months)
        district = np.random.choice(districts)
        zone_type = np.random.choice(commercial_zones)
        industry = np.random.choice(industries)
        
        # 업종과 지역에 따른 기본 매출 범위 설정
        base_revenue = np.random.randint(500000, 5000000)
        
        # 요일별 매출 (주말이 평일보다 높은 경향)
        weekday_multiplier = [0.8, 0.85, 0.9, 0.95, 1.0, 1.3, 1.2]  # 월~일
        daily_revenues = [int(base_revenue * m * np.random.uniform(0.8, 1.2)) 
                         for m in weekday_multiplier]
        
        # 연령대별 매출 비율
        age_distribution = np.random.dirichlet(np.ones(len(age_groups)))
        total_revenue = sum(daily_revenues)
        age_revenues = [int(total_revenue * ratio) for ratio in age_distribution]
        
        # 위도/경도 (서울시 범위)
        latitude = np.random.uniform(37.4, 37.7)
        longitude = np.random.uniform(126.8, 127.2)
        
        record = {
            '기준년월코드': year_month,
            '상권구분코드명': zone_type,
            '상권코드명': f'{district}_{zone_type}_{i%50:03d}',
            '행정구역명': district,
            '서비스업종코드명': industry,
            '월요일매출금액': daily_revenues[0],
            '화요일매출금액': daily_revenues[1],
            '수요일매출금액': daily_revenues[2],
            '목요일매출금액': daily_revenues[3],
            '금요일매출금액': daily_revenues[4],
            '토요일매출금액': daily_revenues[5],
            '일요일매출금액': daily_revenues[6],
            '월매출금액': sum(daily_revenues) * 4,  # 대략 한 달치
            '10대매출금액': age_revenues[0],
            '20대매출금액': age_revenues[1],
            '30대매출금액': age_revenues[2],
            '40대매출금액': age_revenues[3],
            '50대매출금액': age_revenues[4],
            '60대이상매출금액': age_revenues[5],
            '위도': latitude,
            '경도': longitude,
            '점포수': np.random.randint(5, 100)
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    return df

def main():
    """메인 함수: 샘플 데이터 생성 및 저장"""
    
    print("=" * 60)
    print("서울시 상권분석 샘플 데이터 생성")
    print("=" * 60)
    
    # 데이터 생성
    print("\n1. 샘플 데이터 생성 중...")
    df = create_sample_seoul_commercial_data()
    print(f"   ✓ {len(df)}개 레코드 생성 완료")
    
    # 데이터 저장
    output_path = 'data/서울시_상권_추정매출.csv'
    print(f"\n2. 데이터 저장 중: {output_path}")
    df.to_csv(output_path, index=False, encoding='cp949')
    print(f"   ✓ 저장 완료")
    
    # 데이터 미리보기
    print("\n3. 데이터 미리보기:")
    print("-" * 60)
    print(df.head())
    
    print("\n4. 데이터 기본 정보:")
    print("-" * 60)
    print(f"   - 총 레코드 수: {len(df):,}")
    print(f"   - 컬럼 수: {len(df.columns)}")
    print(f"   - 구 개수: {df['행정구역명'].nunique()}")
    print(f"   - 업종 개수: {df['서비스업종코드명'].nunique()}")
    print(f"   - 기간: {df['기준년월코드'].min()} ~ {df['기준년월코드'].max()}")
    
    print("\n" + "=" * 60)
    print("데이터 준비 완료!")
    print("=" * 60)
    print("\n다음 명령어로 데이터를 불러올 수 있습니다:")
    print('df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")')
    print("\n실제 서울시 상권분석 데이터 다운로드:")
    print("https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do")

if __name__ == "__main__":
    main()
