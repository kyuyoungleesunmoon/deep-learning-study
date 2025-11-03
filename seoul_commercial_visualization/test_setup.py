"""
Setup verification script
서울시 상권분석 데이터 시각화 교육 환경 테스트
"""

import sys

def test_imports():
    """필수 패키지 임포트 테스트"""
    print("=" * 60)
    print("1. 필수 패키지 임포트 테스트")
    print("=" * 60)
    
    packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'matplotlib.pyplot': 'plt',
        'seaborn': 'sns',
        'folium': 'folium'
    }
    
    success_count = 0
    for pkg, alias in packages.items():
        try:
            if '.' in pkg:
                exec(f"import {pkg.split('.')[0]}")
            else:
                exec(f"import {pkg}")
            print(f"✓ {pkg} 임포트 성공")
            success_count += 1
        except ImportError as e:
            print(f"✗ {pkg} 임포트 실패: {e}")
    
    print(f"\n결과: {success_count}/{len(packages)} 성공")
    return success_count == len(packages)

def test_data_file():
    """데이터 파일 존재 확인"""
    print("\n" + "=" * 60)
    print("2. 데이터 파일 확인")
    print("=" * 60)
    
    import os
    import pandas as pd
    
    data_path = 'data/서울시_상권_추정매출.csv'
    
    if not os.path.exists(data_path):
        print(f"✗ 데이터 파일 없음: {data_path}")
        print("  → 'python prepare_data.py'를 실행하여 생성하세요.")
        return False
    
    try:
        df = pd.read_csv(data_path, encoding='cp949')
        print(f"✓ 데이터 파일 로드 성공")
        print(f"  - 레코드 수: {len(df):,}개")
        print(f"  - 컬럼 수: {len(df.columns)}개")
        print(f"  - 크기: {os.path.getsize(data_path) / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"✗ 데이터 파일 로드 실패: {e}")
        return False

def test_visualization():
    """기본 시각화 테스트"""
    print("\n" + "=" * 60)
    print("3. 기본 시각화 테스트")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # 한글 폰트 설정 시도
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            print("✓ 한글 폰트 설정 성공 (Malgun Gothic)")
        except:
            print("△ 한글 폰트 설정 실패 (기본 폰트 사용)")
        
        # 간단한 그래프 생성 테스트
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title('테스트 그래프')
        
        # 파일 저장 테스트
        import os
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/test_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        if os.path.exists('output/test_plot.png'):
            print("✓ 그래프 생성 및 저장 성공")
            print("  → output/test_plot.png 생성 완료")
            return True
        else:
            print("✗ 그래프 저장 실패")
            return False
            
    except Exception as e:
        print(f"✗ 시각화 테스트 실패: {e}")
        return False

def test_folium():
    """Folium 지도 생성 테스트"""
    print("\n" + "=" * 60)
    print("4. Folium 지도 생성 테스트")
    print("=" * 60)
    
    try:
        import folium
        
        # 간단한 지도 생성
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
        
        # HTML 저장
        import os
        os.makedirs('output', exist_ok=True)
        m.save('output/test_map.html')
        
        if os.path.exists('output/test_map.html'):
            print("✓ Folium 지도 생성 및 저장 성공")
            print("  → output/test_map.html 생성 완료")
            print("  → 웹 브라우저에서 열어보세요!")
            return True
        else:
            print("✗ 지도 저장 실패")
            return False
            
    except Exception as e:
        print(f"✗ Folium 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  서울시 상권분석 데이터 시각화 교육 - 환경 테스트  ║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    # 테스트 실행
    results.append(("패키지 임포트", test_imports()))
    results.append(("데이터 파일", test_data_file()))
    results.append(("시각화", test_visualization()))
    results.append(("Folium 지도", test_folium()))
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        status = "✓ 통과" if result else "✗ 실패"
        print(f"{test_name:<15}: {status}")
    
    print("-" * 60)
    print(f"총 {success_count}/{total_count} 테스트 통과")
    
    if success_count == total_count:
        print("\n🎉 모든 테스트 통과! 학습을 시작할 준비가 되었습니다.")
        print("\n다음 단계:")
        print("  1. notebooks/1_matplotlib_basic.md 부터 학습 시작")
        print("  2. 각 교재의 코드를 실행하며 실습")
        print("  3. 퀴즈로 학습 내용 확인")
        return 0
    else:
        print("\n⚠️  일부 테스트 실패. 위 오류를 확인하고 해결하세요.")
        print("\n문제 해결:")
        print("  - 패키지 설치: pip install pandas numpy matplotlib seaborn folium")
        print("  - 데이터 생성: python prepare_data.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
