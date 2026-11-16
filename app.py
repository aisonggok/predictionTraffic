import pickle
import numpy as np
import streamlit as st
from Orange.data import Table, Domain

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측",
    page_icon="🚦",
    layout="wide"
)

# 2. Orange3 모델 로드 (캐싱 적용으로 슬라이더 조작 시 빠른 재실행)
@st.cache_resource
def load_orange_model(filepath: str):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("🚨 'traffic_casualty_model.pkcls' 파일을 찾을 수 없습니다. app.py와 같은 경로에 파일을 배치해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 모델 도메인 및 학습 데이터 기반 기본값 추출
orig_domain = model.original_domain
orig_attrs = orig_domain.attributes

# 모델에 내장된 학습 인스턴스(model.instances)에서 각 수치형 변수의 평균값을 계산
default_means = {}
if getattr(model, "instances", None) is not None:
    X_train = model.instances.X
    for idx, attr in enumerate(orig_attrs):
        if attr.is_continuous:
            default_means[attr.name] = float(np.nanmean(X_train[:, idx]))

# '시도' 범주형 목록 추출 (모델에 저장된 값 우선 사용, 없으면 전국 17개 시도 목록 제공)
sido_attr = next((attr for attr in orig_attrs if attr.is_discrete), None)
model_sido_values = list(sido_attr.values) if sido_attr else []
all_sido_options = model_sido_values if len(model_sido_values) >= 10 else list(dict.fromkeys(
    model_sido_values + [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]
))

# 4. 실시간 예측 함수
def predict_casualties(sido_val: str, accident_cnt: int, minor_injury_cnt: int) -> float:
    """
    사용자 입력(시도, 사고건수, 경상자수)을 Orange3 원본 도메인 형식의 Table로 변환하여
    모델 내장 전처리(원-핫 인코딩 및 정규화)를 거쳐 예측값을 산출합니다.
    """
    row = []
    for attr in orig_attrs:
        if attr.is_discrete:
            # 모델 학습 시 포함된 시도명인 경우 해당 인덱스 매핑, 미포함 시 unknown(np.nan) 처리
            if sido_val in attr.values:
                row.append(float(attr.to_val(sido_val)))
            else:
                row.append(np.nan)
        else:
            if "사고건수" in attr.name:
                row.append(float(accident_cnt))
            elif "경상자수" in attr.name:
                row.append(float(minor_injury_cnt))
            else:
                # 슬라이더에 없는 피처(사망자수, 중상자수 등)는 학습 데이터 평균값으로 보정
                row.append(default_means.get(attr.name, 0.0))

    input_domain = Domain(orig_attrs)
    input_table = Table.from_numpy(input_domain, np.array([row], dtype=float))

    # Orange3 모델 예측 실행 (전처리 파이프라인 자동 적용)
    pred = model(input_table)
    return max(0.0, float(pred[0]))


# --- 메인 화면 UI 구성 ---
st.title("🚦 어린이 보호구역 교통사고 위험도 실시간 예측기")
st.markdown("좌측 사이드바에서 **시도, 사고건수, 경상자수**를 조절하면 오렌지3 신경망 모델이 예상 피해 규모와 위험 단계를 실시간으로 계산합니다.")
st.divider()

# 5. 사이드바: 실시간 입력 컨트롤
with st.sidebar:
    st.header("⚙️ 사고 조건 설정")
    selected_sido = st.selectbox("📍 시도 선택", options=all_sido_options, index=0)
    accident_count = st.slider("🚗 사고건수 (건)", min_value=0, max_value=300, value=30, step=1)
    minor_injury_count = st.slider("🩹 경상자수 (명)", min_value=0, max_value=300, value=25, step=1)

    if sido_attr and selected_sido not in sido_attr.values:
        st.caption(f"ℹ️ '{selected_sido}' 지역은 학습 데이터({', '.join(sido_attr.values)}) 외 지역이므로 전체 평균 기준으로 보정 예측됩니다.")

# 6. 예측 실행
predicted_val = predict_casualties(selected_sido, accident_count, minor_injury_count)
target_name = model.domain.class_var.name if model.domain.class_var else "예상 피해자 수"

# 7. 예측 수치에 따른 3단계 신호등 및 안전 수칙 분기
# (모델 타겟인 '사상자수' 예측 분포를 고려한 임계값 설정 — 필요 시 기준 수치 조정 가능)
if predicted_val < 20.0:
    light_emoji = "🟢"
    level_title = "안전 (SAFE)"
    accent_color = "#2ECC71"
    bg_tint = "rgba(46, 204, 113, 0.12)"
    summary_msg = "예상 피해 규모가 낮은 안정 단계입니다. 기본 스쿨존 안전 수칙을 꾸준히 지켜주세요."
    safety_cards = [
        ("시속 30km 이하 서행 유지", "어린이 보호구역 내에서는 전방에 보행자가 없더라도 제한속도 30km/h 이하로 주행합니다."),
        ("횡단보도 앞 일시정지", "신호등이 없는 스쿨존 횡단보도 앞에서는 반드시 일시정지 후 좌우를 살피고 출발합니다."),
        ("어린이 보행 3원칙 지도", "횡단보도를 건너기 전 '일단 멈춤 - 좌우 확인 - 손 들고 건너기' 습관을 안내합니다.")
    ]
elif predicted_val < 50.0:
    light_emoji = "🟡"
    level_title = "주의 (CAUTION)"
    accent_color = "#F1C40F"
    bg_tint = "rgba(241, 196, 15, 0.15)"
    summary_msg = "사고 및 인명 피해 증가 조짐이 감지되었습니다. 시야 확보와 감속 운전이 필수적입니다."
    safety_cards = [
        ("스쿨존 불법 주·정차 절대 금지", "주·정차 차량은 어린이의 키를 가려 운전자의 시야를 차단하는 대형 사고의 주원인입니다."),
        ("교차로 우회전 시 일시정지 및 사각지대 확인", "차량 A필러와 사이드미러 사각지대에 어린이가 있는지 멈춰서 직접 확인 후 회전합니다."),
        ("등·하교 집중 시간대 우회 권장", "어린이 통행량이 급증하는 오전 8~9시, 오후 1~4시에는 가급적 보호구역 외곽 도로를 이용합니다.")
    ]
else:
    light_emoji = "🔴"
    level_title = "위험 (DANGER)"
    accent_color = "#E74C3C"
    bg_tint = "rgba(231, 76, 60, 0.15)"
    summary_msg = "인명 피해 발생 위험이 매우 높은 고위험 경보 상태입니다. 즉각적인 감속과 집중 관리가 필요합니다."
    safety_cards = [
        ("시속 20km 이하 초저속 주행 및 비상등 점등", "어린이의 돌발 횡단 시 즉각 제동할 수 있도록 브레이크에 발을 올린 상태로 서행합니다."),
        ("전방 및 인도 측면 동시 주시", "운전 중 스마트폰 및 내비게이션 조작을 일절 중단하고 인도 위 어린이의 움직임을 예의주시합니다."),
        ("보호구역 안전시설물 긴급 점검", "과속방지턱, 무인교통단속장비, 보행자 방호울타리의 정상 작동 여부를 확인하고 지자체에 통보합니다.")
    ]

# 8. 화면 레이아웃 출력 (좌측: 대형 신호등 / 우측: 예측 결과 및 안전 수칙 카드)
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("🚦 실시간 위험도 신호등")
    st.markdown(
        f"""
        <div style="
            background-color: {bg_tint};
            border: 3px solid {accent_color};
            border-radius: 20px;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        ">
            <div style="font-size: 110px; line-height: 1.1; margin-bottom: 14px;">
                {light_emoji}
            </div>
            <div style="font-size: 34px; font-weight: 800; color: {accent_color}; letter-spacing: 1px;">
                {level_title}
            </div>
            <p style="font-size: 15px; margin-top: 16px; line-height: 1.5;">
                {summary_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_right:
    st.subheader("📊 실시간 예측 결과 및 안전 수칙")

    m1, m2, m3 = st.columns(3)
    m1.metric("선택 시도", selected_sido)
    m2.metric("사고건수 / 경상자수", f"{accident_count}건 / {minor_injury_count}명")
    m3.metric(f"예상 {target_name}", f"{predicted_val:.2f} 명")

    st.divider()
    st.markdown(f"#### 🛡️ {light_emoji} {level_title.split()[0]} 단계 어린이 보호구역 안전 수칙 카드")

    for title, desc in safety_cards:
        with st.container(border=True):
            st.markdown(f"**✅ {title}**")
            st.write(desc)
