import pickle
import streamlit as st
import numpy as np
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable

# 1. 페이지 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측",
    page_icon="🚦",
    layout="wide"
)

# 2. 모델 로드 (캐싱을 통해 슬라이더 조작 시 모델 재로딩 방지)
@st.cache_resource
def load_orange_model(filepath: str):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("🚨 'traffic_casualty_model.pkcls' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 모델 파일을 위치시켜 주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 예측 실행 함수
def predict_fatalities(model, sido_val: str, accident_cnt: int, minor_injury_cnt: int) -> float:
    """
    Orange3 모델의 domain 정보에 맞춰 입력 데이터를 Table 객체로 변환 후 예측값을 반환합니다.
    """
    domain = model.domain
    row_values = []

    for attr in domain.attributes:
        if attr.is_discrete:
            # 범주형 변수(시도) 처리: 모델 학습 시 등록된 카테고리 값과 매칭
            if sido_val in attr.values:
                row_values.append(attr.to_val(sido_val))
            else:
                # 모델의 시도 표기법과 다를 경우(예: '서울' vs '서울특별시') 부분 일치 탐색
                matched = [v for v in attr.values if sido_val in v or v in sido_val]
                row_values.append(attr.to_val(matched[0]) if matched else 0)
        else:
            # 연속형 변수(사고건수, 경상자수) 처리
            if "사고" in attr.name:
                row_values.append(float(accident_cnt))
            elif "경상" in attr.name:
                row_values.append(float(minor_injury_cnt))
            else:
                row_values.append(0.0)

    # Orange3 Table 생성 (특성 변수만 포함한 입력용 Domain 구성)
    input_domain = Domain(domain.attributes)
    input_table = Table.from_numpy(input_domain, np.array([row_values], dtype=float))

    # 예측 실행 (회귀 모델 기준 float 반환)
    prediction = model(input_table)
    pred_value = float(prediction[0])
    return max(0.0, pred_value)  # 음수 예측 방지


# --- UI 레이아웃 구성 ---
st.title("🚦 어린이 보호구역 교통사고 위험도 실시간 시뮬레이터")
st.markdown("좌측 사이드바에서 **시도, 사고건수, 경상자수**를 조절하면 오렌지3 AI 모델이 **예상 사망자 수**와 **위험 등급**을 실시간으로 분석합니다.")
st.divider()

# 4. 사이드바: 사용자 입력 컨트롤 (변경 즉시 자동 재실행됨)
with st.sidebar:
    st.header("⚙️ 사고 조건 입력")

    # 모델에 저장된 실제 '시도' 카테고리 목록이 있으면 우선 사용, 없으면 기본 리스트 사용
    default_sido_list = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]
    discrete_attrs = [attr for attr in model.domain.attributes if attr.is_discrete]
    sido_options = list(discrete_attrs[0].values) if discrete_attrs else default_sido_list

    selected_sido = st.selectbox("📍 시도 선택", options=sido_options, index=0)
    accident_count = st.slider("🚗 사고건수 (건)", min_value=0, max_value=500, value=30, step=1)
    minor_injury_count = st.slider("🩹 경상자수 (명)", min_value=0, max_value=500, value=25, step=1)

# 5. 실시간 예측 계산
predicted_deaths = predict_fatalities(model, selected_sido, accident_count, minor_injury_count)

# 6. 위험도 임계값 판별 (데이터 특성에 맞게 기준 수치 조정 가능)
if predicted_deaths < 0.5:
    status_icon = "🟢"
    status_label = "안전 (SAFE)"
    status_color = "#2ECC71"
    bg_color = "rgba(46, 204, 113, 0.12)"
    status_desc = "예상 사망자 수가 매우 낮은 안정적인 수준입니다. 현재의 보호구역 안전 수칙을 지속적으로 유지하세요."
    safety_tips = [
        "**규정 속도 준수:** 어린이 보호구역 내 시속 30km 이하 서행 운전을 생활화합니다.",
        "**횡단보도 일시정지:** 보행자가 없더라도 횡단보도 앞에서는 반드시 일시정지 후 출발합니다.",
        "**어린이 방어보행 지도:** 횡단보도를 건너기 전 '멈춘다 - 살핀다 - 건넌다' 3원칙을 안내합니다."
    ]
elif predicted_deaths < 1.5:
    status_icon = "🟡"
    status_label = "주의 (CAUTION)"
    status_color = "#F1C40F"
    bg_color = "rgba(241, 196, 15, 0.15)"
    status_desc = "사고 규모 대비 중대 인명 피해 발생 가능성이 감지되었습니다. 주변 시야 확보와 감속이 필수적입니다."
    safety_tips = [
        "**불법 주·정차 절대 금지:** 주·정차 차량은 어린이의 시야를 가리는 '스쿨존 사고'의 주요 원인입니다.",
        "**사각지대 확인:** 우회전 및 교차로 진입 시 차량 A필러 사각지대와 사이드미러를 철저히 확인합니다.",
        "**등·하교 시간대 집중 경계:** 어린이 통행이 집중되는 오전 8~9시, 오후 1~4시에는 우회 도로 이용을 권장합니다."
    ]
else:
    status_icon = "🔴"
    status_label = "위험 (DANGER)"
    status_color = "#E74C3C"
    bg_color = "rgba(231, 76, 60, 0.15)"
    status_desc = "사망 사고 발생 위험이 매우 높은 고위험 상태입니다. 즉각적인 교통 통제 및 집중 단속이 요구됩니다."
    safety_tips = [
        "**즉시 감속 및 비상등 점등:** 시속 20km 이하로 서행하며 돌발 횡단 상황에 즉각 제동할 수 있도록 대비합니다.",
        "**스마트폰 및 전방 주시 태만 금지:** 운전 중 기기 조작을 멈추고 전방과 인도 측면을 동시에 주시합니다.",
        "**안전시설물 점검 요청:** 과속방지턱, 무인단속카메라, 방호울타리 파손 여부를 관할 지자체에 즉시 신고합니다."
    ]

# 7. 메인 화면 출력 (좌측: 대형 신호등 카드 / 우측: 예측 수치 및 안전 수칙 카드)
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("🚦 실시간 위험도 신호등")
    # 대형 신호등 및 등급 표시 HTML 카드
    traffic_light_html = f"""
    <div style="
        background-color: {bg_color};
        border: 3px solid {status_color};
        border-radius: 20px;
        padding: 35px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    ">
        <div style="font-size: 95px; line-height: 1.1; margin-bottom: 12px;">
            {status_icon}
        </div>
        <div style="font-size: 32px; font-weight: 800; color: {status_color}; letter-spacing: 1px;">
            {status_label}
        </div>
        <p style="font-size: 15px; margin-top: 14px; opacity: 0.9;">
            {status_desc}
        </p>
    </div>
    """
    st.markdown(traffic_light_html, unsafe_allow_html=True)

with col2:
    st.subheader("📊 AI 모델 예측 결과 및 안전 수칙")

    # 요약 지표 출력
    m1, m2, m3 = st.columns(3)
    m1.metric(label="선택 지역", value=selected_sido)
    m2.metric(label="사고 / 경상자", value=f"{accident_count}건 / {minor_injury_count}명")
    m3.metric(label="예상 사망자 수", value=f"{predicted_deaths:.2f} 명")

    st.markdown("---")

    # 어린이 보호구역 안전 수칙 카드
    st.markdown(f"#### 🛡️ [{status_label.split()[0]}] 단계 어린이 보호구역 필수 안전 수칙")
    for idx, tip in enumerate(safety_tips, 1):
        st.info(f"**{idx}.** {tip}")