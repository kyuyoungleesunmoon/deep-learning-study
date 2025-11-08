"""
7교시: 종합 프로젝트 실행 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("📂 데이터 로드 중...")
    df = pd.read_csv('../data/sales_data.csv')
    
    # 날짜 변환
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # 결측치 처리
    df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
    df['region'].fillna('Unknown', inplace=True)
    
    # 특성 생성
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    
    return df

def build_prediction_model(df):
    """예측 모델 구축"""
    print("\n🤖 모델 구축 중...")
    
    # 인코딩
    le_cat = LabelEncoder()
    df['category_encoded'] = le_cat.fit_transform(df['product_category'])
    
    # 특성 선택
    X = df[['quantity', 'unit_price', 'customer_age', 'category_encoded', 'month']]
    y = df['final_amount']
    
    # 분리 및 스케일링
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 평가
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ RMSE: {rmse:,.0f}원")
    
    return model, scaler

def main():
    print("="*60)
    print("🚀 온라인 쇼핑몰 데이터 분석 프로젝트")
    print("="*60)
    
    # 데이터 준비
    df = load_and_prepare_data()
    
    # 기본 통계
    print(f"\n💰 총 매출: {df['final_amount'].sum():,.0f}원")
    print(f"📦 총 주문: {len(df):,}건")
    print(f"👥 고유 고객: {df['customer_id'].nunique():,}명")
    
    # 카테고리 분석
    print("\n🏷️ 카테고리별 매출:")
    category_sales = df.groupby('product_category')['final_amount'].sum().sort_values(ascending=False)
    print(category_sales)
    
    # 모델 구축
    model, scaler = build_prediction_model(df)
    
    print("\n🎉 프로젝트 완료!")

if __name__ == "__main__":
    main()
