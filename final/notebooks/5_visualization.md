# 5교시: 데이터 시각화 - Matplotlib & Seaborn

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐⭐  
> **목표**: 효과적인 데이터 시각화 방법을 익히고 실무 보고서를 작성합니다.

---

## 📚 학습 내용

1. Matplotlib 기초
2. Seaborn 시각화
3. 실무 시각화 패턴

---

## 1. Matplotlib 기초

### 1.1 라인 차트

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 준비
df = pd.read_csv('../data/sales_data.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')

# 월별 매출 추이
monthly_sales = df.groupby('month')['final_amount'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].astype(str)

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['month'], monthly_sales['final_amount'], marker='o', linewidth=2)
plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Sales Amount (KRW)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/monthly_trend.png', dpi=300)
plt.show()
```

### 1.2 막대 차트

```python
# 카테고리별 매출
category_sales = df.groupby('product_category')['final_amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(category_sales.index, category_sales.values, color='steelblue')
plt.title('Sales by Category', fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales (KRW)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../output/figures/category_sales.png', dpi=300)
plt.show()
```

### 1.3 산점도

```python
# 수량 vs 매출액
plt.figure(figsize=(10, 6))
plt.scatter(df['quantity'], df['final_amount'], alpha=0.5)
plt.title('Quantity vs Sales Amount', fontsize=16)
plt.xlabel('Quantity', fontsize=12)
plt.ylabel('Sales Amount (KRW)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/scatter_plot.png', dpi=300)
plt.show()
```

### 1.4 히스토그램

```python
# 매출액 분포
plt.figure(figsize=(10, 6))
plt.hist(df['final_amount'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales Amount', fontsize=16)
plt.xlabel('Sales Amount (KRW)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['final_amount'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['final_amount'].median(), color='green', linestyle='--', label='Median')
plt.legend()
plt.tight_layout()
plt.savefig('../output/figures/sales_distribution.png', dpi=300)
plt.show()
```

---

## 2. Seaborn 시각화

### 2.1 Box Plot

```python
# 카테고리별 매출 분포
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='product_category', y='final_amount')
plt.title('Sales Distribution by Category', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../output/figures/boxplot_category.png', dpi=300)
plt.show()
```

### 2.2 상관관계 히트맵

```python
# 수치형 컬럼만 선택
numeric_cols = ['quantity', 'unit_price', 'total_amount', 'discount_rate', 'final_amount', 'customer_age']
correlation = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('../output/figures/correlation_heatmap.png', dpi=300)
plt.show()
```

### 2.3 Count Plot

```python
# 결제 수단별 거래 건수
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='payment_method', order=df['payment_method'].value_counts().index)
plt.title('Transaction Count by Payment Method', fontsize=16)
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('../output/figures/payment_count.png', dpi=300)
plt.show()
```

### 2.4 Pair Plot

```python
# 다변량 관계 탐색 (샘플링하여 시각화)
sample_df = df[['quantity', 'unit_price', 'final_amount', 'customer_age']].sample(500)
sns.pairplot(sample_df)
plt.savefig('../output/figures/pairplot.png', dpi=300)
plt.show()
```

---

## 3. 서브플롯

```python
# 2x2 그리드
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 월별 매출
monthly_sales = df.groupby('month')['final_amount'].sum()
axes[0, 0].plot(range(len(monthly_sales)), monthly_sales.values, marker='o')
axes[0, 0].set_title('Monthly Sales Trend')
axes[0, 0].grid(True, alpha=0.3)

# 2. 카테고리별 매출
category_sales = df.groupby('product_category')['final_amount'].sum().sort_values()
axes[0, 1].barh(category_sales.index, category_sales.values)
axes[0, 1].set_title('Sales by Category')

# 3. 매출 분포
axes[1, 0].hist(df['final_amount'], bins=50, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Sales Amount Distribution')
axes[1, 0].set_xlabel('Amount')

# 4. Box plot
df.boxplot(column='final_amount', by='product_category', ax=axes[1, 1])
axes[1, 1].set_title('Sales Distribution by Category')
axes[1, 1].set_xlabel('')

plt.tight_layout()
plt.savefig('../output/figures/dashboard.png', dpi=300)
plt.show()
```

---

## 💪 실습 문제

### 문제 1: 지역별 매출 비교 시각화

```python
# TODO: 지역별 총 매출을 막대 차트로 시각화
```

### 문제 2: 연령대별 구매 패턴

```python
# TODO: 연령대별 평균 구매액을 시각화
```

---

**수고하셨습니다! 🎉**
