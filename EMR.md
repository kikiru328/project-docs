# EMR 데이터 기반 질환 예측 모델 개발

> ⚠️ *해당 프로젝트는 기관 요청에 따라 **주제** 및 **코드** 가 비공개(private) 처리되어 있으며, 구현 세부 내용은 README로만 제공됩니다.*

비정형 EMR 데이터를 정제하고 통계 분석 및 머신러닝 예측 모델을 개발하였습니다.  
기존 모델 대비 ROC 성능을 61% → 87%까지 약 45% 향상시킨 프로젝트입니다.  
전체 데이터 처리 파이프라인과 학습 스크립트를 자동화하여 **생산성을 60% 이상 개선**하였습니다.  
이로서 결과는 **2편의 논문**으로 작성되었습니다.

## 개요

- 기간: 2023.03 ~ 2024.04
- 역할: 데이터 정제, 변수 재구성, 모델 개발, 성능 보고서 작성 (단독 수행)
- 기술 스택: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, SHAPs, SQL, Scipy

## 문제 정의

| 문제 | 설명 |
|------|------|
| 비정형 수기 EMR 데이터 | 단위/표현 불일치, 극단값 존재 |
| 변수 간 약한 상관성 | 예측에 기여하는 변수 재설계 필요 |
| 반복 실험 비효율 | 파이프라인 구조 부재로 자동화 어려움

## 해결방안

- 비정형 수기 EMR 데이터
    - 기록 단위를 일치시키고, 문자열 삭제, 수치화로 변경
    - 극단값의 경우 `사분위수`로 1차 처리 (사람의 논리로서 벗어난 이상치 (ex: 키=1602cm 등))  
    - 질환의 특징일 수도 있기에 2차로 의료진 및 데이터관리 팀에게 문의.
- 변수 간 약한 상관성
    - 관련 연구 논문 및 질환 관련 논문 참고하여 파생변수 생성
    - 통계 분석 기반 변수 선택
- 반복 실험 비효율
    - 학습~검증~리포트 생성을 포함한 "pipeline script" 개발하여 자동화


## 모델 성능 비교
![Image](https://github.com/user-attachments/assets/dbe09730-4d48-4e75-8bd2-701068f3faa9)

| 모델 | ROC Score |
|------|-----------|
| 관련 논문 | 61% |
| 초기 학습 | 45% |
| 최종 모델 | **87%** |

## 성과
| 항목 | 결과 |
|------|-------|
| 전처리 시간 | 60% 단축 |
| 모델 성능 | 45% 향상 (ROC 기준) |
| 연구 산출물 | 논문 2편 작성 (성과 기반)
