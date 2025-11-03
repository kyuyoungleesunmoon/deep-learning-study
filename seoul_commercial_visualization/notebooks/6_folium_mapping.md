# 6교시: Folium - 지도 위 상권 위치 시각화

**학습 목표:**
- Folium 라이브러리를 활용한 인터랙티브 지도 생성
- 마커와 팝업으로 상권 정보 표시
- 마커 클러스터링으로 밀집도 시각화
- 실무 의사결정을 위한 지리공간 분석

**예상 소요 시간:** 60분

---

## 1. Folium 소개

### 1.1 Folium이란?

**Folium**은 Python에서 인터랙티브한 지도를 쉽게 만들 수 있는 라이브러리입니다.
- **기반**: Leaflet.js (JavaScript 지도 라이브러리)
- **특징**: 확대/축소, 드래그, 클릭 등 인터랙티브 기능
- **용도**: 매장 위치, 배달 구역, 부동산 분석 등
- **장점**: 코드 몇 줄로 전문가 수준의 지도 생성

**Matplotlib/Seaborn vs Folium:**
- Matplotlib/Seaborn: 정적 이미지 (보고서용)
- Folium: 인터랙티브 HTML (웹, 프레젠테이션용)

```python
import pandas as pd
import numpy as np
import folium
from folium import plugins

# 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")

print("=" * 60)
print("Folium 지도 시각화 시작")
print("=" * 60)
print(f"총 데이터: {len(df)}개")
print(f"위도 범위: {df['위도'].min():.4f} ~ {df['위도'].max():.4f}")
print(f"경도 범위: {df['경도'].min():.4f} ~ {df['경도'].max():.4f}")
```

---

## 2. 기본 지도 생성

### 2.1 서울 중심 지도 만들기

```python
# 서울 중심 좌표
seoul_center = [37.5665, 126.9780]  # 시청 근처

# 기본 지도 생성
m = folium.Map(
    location=seoul_center,
    zoom_start=11,  # 확대 수준 (1=세계, 18=건물 수준)
    tiles='OpenStreetMap'  # 지도 스타일
)

# HTML 파일로 저장
m.save('output/6_basic_map.html')
print("\n기본 지도 생성 완료: output/6_basic_map.html")
print("웹 브라우저에서 열어보세요!")
```

### 2.2 다양한 지도 스타일

```python
# 여러 타일 스타일 비교
tile_styles = {
    'OpenStreetMap': '기본 (상세한 거리 정보)',
    'Stamen Terrain': '지형 (산, 강 등)',
    'Stamen Toner': '흑백 (미니멀)',
    'CartoDB positron': '밝은 배경 (데이터 강조)',
    'CartoDB dark_matter': '어두운 배경 (야간 모드)'
}

print("\n" + "=" * 60)
print("Folium 지도 타일 스타일")
print("=" * 60)
for style, description in tile_styles.items():
    print(f"- {style}: {description}")
```

---

## 3. 마커 추가: 상권 위치 표시

### 3.1 단일 마커 추가

```python
# 강남역 근처 커피숍 1개 샘플
sample = df[df['서비스업종코드명'] == '커피/음료'].iloc[0]

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=12)

# 마커 추가
folium.Marker(
    location=[sample['위도'], sample['경도']],
    popup=f"""
    <b>{sample['서비스업종코드명']}</b><br>
    상권: {sample['상권코드명']}<br>
    지역: {sample['행정구역명']}<br>
    월매출: {sample['월매출금액']/1000000:.2f}M원
    """,
    tooltip='클릭하여 상세 정보 보기',
    icon=folium.Icon(color='blue', icon='info-sign')
).add_to(m)

m.save('output/6_single_marker_map.html')
print("\n단일 마커 지도 생성: output/6_single_marker_map.html")
```

### 3.2 여러 마커 추가 (업종별)

```python
# 강남구의 주요 업종 상위 20개 매장
gangnam_data = df[df['행정구역명'] == '강남구'].nlargest(20, '월매출금액')

# 지도 생성
m = folium.Map(location=[37.5172, 127.0473], zoom_start=13)  # 강남구 중심

# 업종별 아이콘 색상
industry_colors = {
    '커피/음료': 'blue',
    '한식음식점': 'green',
    '편의점': 'orange',
    '치킨전문점': 'red',
    '의류': 'purple'
}

# 마커 추가
for idx, row in gangnam_data.iterrows():
    color = industry_colors.get(row['서비스업종코드명'], 'gray')
    
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=f"""
        <b>{row['서비스업종코드명']}</b><br>
        상권: {row['상권코드명']}<br>
        월매출: {row['월매출금액']/1000000:.2f}M원<br>
        점포수: {row['점포수']}개
        """,
        tooltip=row['서비스업종코드명'],
        icon=folium.Icon(color=color, icon='shop')
    ).add_to(m)

# 범례 추가 (HTML)
legend_html = '''
<div style="position: fixed; 
            top: 10px; right: 10px; width: 200px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<b>업종별 마커 색상</b><br>
<i class="fa fa-map-marker" style="color:blue"></i> 커피/음료<br>
<i class="fa fa-map-marker" style="color:green"></i> 한식음식점<br>
<i class="fa fa-map-marker" style="color:orange"></i> 편의점<br>
<i class="fa fa-map-marker" style="color:red"></i> 치킨전문점<br>
<i class="fa fa-map-marker" style="color:purple"></i> 의류<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save('output/6_multi_marker_map.html')
print("업종별 마커 지도 생성: output/6_multi_marker_map.html")
```

---

## 4. 마커 클러스터링: 밀집도 시각화

### 4.1 MarkerCluster 사용

```python
# 서울시 전체 커피/음료 매장
coffee_data = df[df['서비스업종코드명'] == '커피/음료']

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=11)

# 마커 클러스터 생성
marker_cluster = plugins.MarkerCluster().add_to(m)

# 모든 커피숍 마커 추가
for idx, row in coffee_data.iterrows():
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=f"""
        <b>{row['상권코드명']}</b><br>
        지역: {row['행정구역명']}<br>
        월매출: {row['월매출금액']/1000000:.2f}M원
        """,
        tooltip='커피/음료',
        icon=folium.Icon(color='blue', icon='coffee', prefix='fa')
    ).add_to(marker_cluster)

m.save('output/6_marker_cluster_map.html')
print("\n마커 클러스터 지도 생성: output/6_marker_cluster_map.html")
print("확대하면 클러스터가 개별 마커로 분리됩니다!")
```

### 4.2 결과 해석

**실무 인사이트:**
- **큰 숫자 클러스터**: 매장이 밀집된 지역 (경쟁 심화)
- **작은 숫자 클러스터**: 매장이 적은 지역 (블루오션 또는 수요 부족)
- **확대/축소**: 지역별 밀집도를 인터랙티브하게 탐색

---

## 5. 히트맵: 밀집도 시각화

### 5.1 HeatMap 생성

```python
# 커피/음료 매장 위치 데이터
coffee_locations = coffee_data[['위도', '경도']].values.tolist()

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=11)

# 히트맵 추가
plugins.HeatMap(
    coffee_locations,
    radius=15,  # 반경
    blur=25,  # 흐림 정도
    max_zoom=13,
    gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
).add_to(m)

m.save('output/6_heatmap.html')
print("\n히트맵 생성: output/6_heatmap.html")
```

### 5.2 가중 히트맵 (매출 기반)

```python
# 매출 기반 가중치
coffee_data_weighted = coffee_data.copy()
coffee_data_weighted['weight'] = coffee_data_weighted['월매출금액'] / coffee_data_weighted['월매출금액'].max()

# 위치 + 가중치
weighted_locations = coffee_data_weighted[['위도', '경도', 'weight']].values.tolist()

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=11)

# 가중 히트맵
plugins.HeatMap(
    weighted_locations,
    radius=20,
    blur=30,
    max_zoom=13,
    gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
).add_to(m)

# 제목 추가
title_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; width: 400px; height: 50px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:16px; padding: 10px; text-align: center">
<b>커피/음료 매출 히트맵</b><br>
빨강: 고매출 밀집, 파랑: 저매출 밀집
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

m.save('output/6_weighted_heatmap.html')
print("가중 히트맵 생성: output/6_weighted_heatmap.html")
```

---

## 6. 원형 마커: 매출 크기 표현

### 6.1 CircleMarker로 매출 크기 시각화

```python
# 강남구, 마포구, 종로구의 고매출 매장
selected_districts = ['강남구', '마포구', '종로구']
high_revenue = df[df['행정구역명'].isin(selected_districts)].nlargest(50, '월매출금액')

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=12)

# CircleMarker 추가
for idx, row in high_revenue.iterrows():
    # 매출에 비례하는 반경 (스케일링)
    radius = (row['월매출금액'] / high_revenue['월매출금액'].max()) * 30 + 5
    
    # 색상 (업종별)
    color = 'red' if row['월매출금액'] > high_revenue['월매출금액'].median() else 'blue'
    
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=radius,
        popup=f"""
        <b>{row['서비스업종코드명']}</b><br>
        {row['행정구역명']}<br>
        월매출: {row['월매출금액']/1000000:.2f}M원
        """,
        tooltip=f"{row['월매출금액']/1000000:.1f}M원",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.6
    ).add_to(m)

m.save('output/6_circle_marker_map.html')
print("\n원형 마커 지도 생성: output/6_circle_marker_map.html")
```

---

## 7. 다각형 (Polygon): 구역 경계

### 7.1 구별 평균 매출 색상 표현

```python
# 주의: 실제 구 경계 데이터는 별도 GeoJSON 필요
# 여기서는 개념 설명용 예제

# 구별 평균 매출 계산
district_avg = df.groupby('행정구역명')['월매출금액'].mean().sort_values(ascending=False)

print("\n" + "=" * 60)
print("구별 평균 월매출 Top 10")
print("=" * 60)
for idx, (district, avg_sales) in enumerate(district_avg.head(10).items(), 1):
    print(f"{idx}. {district}: {avg_sales/1000000:.2f}M원")

print("\n** 주의 **")
print("실제 구 경계를 지도에 표시하려면 GeoJSON 파일이 필요합니다.")
print("서울시 열린데이터광장에서 'SIG_경계' 데이터를 다운로드할 수 있습니다.")
```

---

## 8. 실무 적용 예시

### 시나리오: 신규 카페 최적 입점 위치 찾기

```python
# 1단계: 커피/음료 경쟁 현황 파악
coffee_all = df[df['서비스업종코드명'] == '커피/음료']

# 2단계: 고매출 지역 식별
high_revenue_coffee = coffee_all[coffee_all['월매출금액'] >= coffee_all['월매출금액'].quantile(0.75)]

# 3단계: 지도 생성
m = folium.Map(location=seoul_center, zoom_start=12)

# 기존 커피숍 (경쟁자) - 작은 파란 원
for idx, row in coffee_all.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,
        color='blue',
        fill=True,
        fillColor='blue',
        fillOpacity=0.3,
        popup='기존 커피숍'
    ).add_to(m)

# 고매출 커피숍 (벤치마크 대상) - 큰 빨간 원
for idx, row in high_revenue_coffee.iterrows():
    radius = (row['월매출금액'] / coffee_all['월매출금액'].max()) * 20 + 5
    
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=radius,
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.7,
        popup=f"""
        <b>고매출 커피숍</b><br>
        지역: {row['행정구역명']}<br>
        월매출: {row['월매출금액']/1000000:.2f}M원<br>
        점포수: {row['점포수']}개
        """
    ).add_to(m)

# 입점 후보지 (예시: 강남역, 홍대입구, 신촌)
candidates = [
    {'name': '강남역', 'lat': 37.4979, 'lon': 127.0276},
    {'name': '홍대입구', 'lat': 37.5572, 'lon': 126.9239},
    {'name': '신촌', 'lat': 37.5559, 'lon': 126.9364}
]

for candidate in candidates:
    folium.Marker(
        location=[candidate['lat'], candidate['lon']],
        popup=f"<b>입점 후보지: {candidate['name']}</b>",
        tooltip=candidate['name'],
        icon=folium.Icon(color='green', icon='star', prefix='fa')
    ).add_to(m)

# 범례
legend_html = '''
<div style="position: fixed; 
            top: 10px; right: 10px; width: 250px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<b>신규 카페 입점 분석</b><br><br>
<i class="fa fa-circle" style="color:blue"></i> 기존 커피숍 (경쟁자)<br>
<i class="fa fa-circle" style="color:red"></i> 고매출 커피숍 (벤치마크)<br>
<i class="fa fa-star" style="color:green"></i> 입점 후보지<br><br>
<b>분석 포인트:</b><br>
- 빨간 원 근처: 수요 높음, 경쟁 심화<br>
- 파란 원 적은 곳: 블루오션 가능<br>
- 고려사항: 유동인구, 임대료 등
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save('output/6_location_analysis_map.html')
print("\n입점 위치 분석 지도 생성: output/6_location_analysis_map.html")

# 후보지별 분석 리포트
print("\n" + "=" * 70)
print("입점 후보지 분석 리포트")
print("=" * 70)

for candidate in candidates:
    # 후보지 주변 1km 내 커피숍 수 (대략적 계산)
    from math import radians, sin, cos, sqrt, atan2
    
    def calculate_distance(lat1, lon1, lat2, lon2):
        """두 좌표 간 거리 계산 (km)"""
        R = 6371  # 지구 반경 (km)
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    # 1km 내 커피숍
    nearby_shops = []
    for idx, row in coffee_all.iterrows():
        distance = calculate_distance(candidate['lat'], candidate['lon'], 
                                     row['위도'], row['경도'])
        if distance <= 1.0:  # 1km 이내
            nearby_shops.append({
                'distance': distance,
                'revenue': row['월매출금액']
            })
    
    avg_revenue = np.mean([s['revenue'] for s in nearby_shops]) if nearby_shops else 0
    
    print(f"\n[{candidate['name']}]")
    print(f"  1km 내 커피숍 수: {len(nearby_shops)}개")
    print(f"  주변 평균 매출: {avg_revenue/1000000:.2f}M원")
    print(f"  경쟁 강도: ", end='')
    
    if len(nearby_shops) > 15:
        print("높음 (레드오션)")
    elif len(nearby_shops) > 8:
        print("중간 (적정)")
    else:
        print("낮음 (블루오션 또는 수요 부족)")
```

---

## 9. 추가 기능: 레이어 컨트롤

### 9.1 여러 레이어를 토글할 수 있는 지도

```python
# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=12)

# 레이어 1: 커피/음료
coffee_layer = folium.FeatureGroup(name='커피/음료')
for idx, row in coffee_all.head(30).iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color='blue',
        fill=True,
        fillOpacity=0.6,
        popup='커피/음료'
    ).add_to(coffee_layer)
coffee_layer.add_to(m)

# 레이어 2: 한식음식점
korean_data = df[df['서비스업종코드명'] == '한식음식점']
korean_layer = folium.FeatureGroup(name='한식음식점')
for idx, row in korean_data.head(30).iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color='green',
        fill=True,
        fillOpacity=0.6,
        popup='한식음식점'
    ).add_to(korean_layer)
korean_layer.add_to(m)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(m)

m.save('output/6_layer_control_map.html')
print("\n레이어 컨트롤 지도 생성: output/6_layer_control_map.html")
print("우측 상단에서 레이어를 켜고 끌 수 있습니다!")
```

---

## 10. 학습 정리

### 10.1 핵심 개념

1. **Folium**: 인터랙티브 지도 생성 라이브러리
2. **Marker**: 특정 위치 표시 (색상, 아이콘 커스터마이징)
3. **MarkerCluster**: 밀집된 마커를 그룹화
4. **HeatMap**: 밀집도를 색상으로 표현
5. **CircleMarker**: 크기로 값의 크기 표현
6. **LayerControl**: 여러 데이터 레이어 토글

### 10.2 실무 활용 팁

- **확대/축소 레벨**: zoom_start로 초기 뷰 조정
- **팝업 HTML**: HTML 태그로 풍부한 정보 제공
- **색상 코딩**: 카테고리나 값의 크기에 따라 색상 구분
- **파일 저장**: `.save()`로 HTML 파일 생성 → 공유 가능
- **웹 배포**: 생성된 HTML을 웹서버에 업로드하여 공유

---

## 11. 학습 확인 퀴즈

### 문제 1
Folium 지도의 가장 큰 장점은?
1. 정적 이미지를 빠르게 생성
2. 확대/축소/드래그 등 인터랙티브 기능
3. 3D 시각화
4. 애니메이션 생성

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 확대/축소/드래그 등 인터랙티브 기능**

**해설:** 
Folium은 인터랙티브한 웹 지도를 생성하는 것이 주요 목적입니다. 사용자가 지도를 클릭하고, 확대/축소하고, 드래그하여 탐색할 수 있습니다. 생성된 HTML 파일은 웹 브라우저에서 바로 열 수 있어 발표나 보고서에 효과적입니다.
</details>

---

### 문제 2
MarkerCluster의 주요 용도는?
1. 마커 색상을 변경
2. 밀집된 마커를 자동으로 그룹화하여 가독성 향상
3. 마커를 삭제
4. 마커 크기를 조정

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 밀집된 마커를 자동으로 그룹화하여 가독성 향상**

**해설:** 
MarkerCluster는 가까이 있는 마커들을 자동으로 그룹화하여 숫자로 표시합니다. 확대하면 클러스터가 개별 마커로 분리됩니다. 이를 통해 수백~수천 개의 마커를 효과적으로 시각화할 수 있으며, 어느 지역에 매장이 밀집되어 있는지 한눈에 파악할 수 있습니다.
</details>

---

### 문제 3
HeatMap을 사용하기에 가장 적합한 경우는?
1. 정확한 매장 위치 표시
2. 업종별 색상 구분
3. 지역의 밀집도나 활동도 시각화
4. 경로 표시

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 지역의 밀집도나 활동도 시각화**

**해설:** 
HeatMap은 데이터의 밀집도를 색상(파랑→초록→노랑→빨강)으로 표현합니다. 
- 빨간색 지역: 매장이 많거나 매출이 높음
- 파란색 지역: 매장이 적거나 매출이 낮음

정확한 위치보다는 전반적인 패턴과 핫스팟을 파악하는 데 적합합니다.
예: 유동인구 분석, 범죄 발생 밀집도, 배달 주문 밀집도
</details>

---

### 문제 4
CircleMarker의 반경을 매출에 비례하여 설정하는 이유는?
1. 그래프를 예쁘게 만들기 위해
2. 값의 크기를 시각적으로 직관적으로 표현하기 위해
3. 마커를 숨기기 위해
4. 색상을 변경하기 위해

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 값의 크기를 시각적으로 직관적으로 표현하기 위해**

**해설:** 
CircleMarker의 반경을 데이터 값에 비례하도록 설정하면, 원의 크기만으로도 값의 상대적 크기를 파악할 수 있습니다.
- 큰 원: 높은 매출
- 작은 원: 낮은 매출

이는 비례 심볼 지도(proportional symbol map)의 기본 원리이며, 실무에서 지역별 차이를 강조할 때 매우 효과적입니다.
</details>

---

### 문제 5
Folium 지도를 다른 사람과 공유하는 가장 쉬운 방법은?
1. 코드 파일(.py) 공유
2. 생성된 HTML 파일 공유
3. 데이터 파일(.csv) 공유
4. 스크린샷 공유

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 생성된 HTML 파일 공유**

**해설:** 
Folium의 `.save()` 메서드는 독립적인 HTML 파일을 생성합니다. 이 파일은:
- Python이 설치되지 않은 환경에서도 웹 브라우저로 열 수 있음
- 인터랙티브 기능(확대/축소/클릭)이 모두 작동
- 이메일, 클라우드, 웹서버 등으로 쉽게 공유 가능

발표나 보고서에 첨부하거나, 웹사이트에 임베드할 수도 있어 매우 실용적입니다.
</details>

---

## 다음 교시 예고

**7교시: 종합 프로젝트 - 가설 수립부터 발표까지** (11/8 토, 1~7교시)
- 프로젝트 주제 선정 및 가설 수립
- 데이터 전처리 및 EDA
- 핵심 인사이트 도출
- 시각화 및 스토리텔링
- 분석 보고서 작성
- 발표 준비 및 발표

**학습을 마치며:**
지금까지 Matplotlib, Seaborn, Folium을 활용한 데이터 시각화 기법을 배웠습니다.
다음 시간에는 이 모든 기술을 종합하여 실전 프로젝트를 수행합니다!
